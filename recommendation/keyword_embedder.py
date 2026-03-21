"""
keyword_embedder.py
-------------------
Embedding-based mapping from Airbnb TF-IDF keywords to:
  - target emotion label(s)   via provided metadata (if available) or zero-shot NLI
  - target audio feature vector via retrieve-then-aggregate over the track catalog

Approach
--------
1. Embed each input keyword with sentence-transformers (all-MiniLM-L6-v2).

2. EMOTION — Metadata or Zero-shot NLI:
   Use provided emotion metadata (dominant_emotion / emotion_scores) if available,
   otherwise classify the concatenated keyword string against the emotion labels
   using a zero-shot NLI pipeline (facebook/bart-large-mnli).

3. AUDIO — Retrieve-then-aggregate:
   Find the top-k tracks in the catalog whose lyric embeddings are most similar
   to the mean keyword embedding (cosine similarity). Average their z-scored
   audio features to form the target vector.

4. LOCATION TERMS:
   Keywords whose NLI confidence across all emotion labels is below a threshold
   are treated as location/place terms and passed to the lyrics-search layer.

Both models are loaded once and cached at module level.
The public API (resolve_keywords return shape) is unchanged so recommender.py
requires no modifications.

Index building
--------------
Call `build_lyric_index(filtered_df, scaled_df, feature_cols)` once after
loading the dataset to pre-embed all lyrics and cache the index. If the index
has not been built, resolve_keywords falls back to returning zero audio targets.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm
from transformers import pipeline

# ── Sentence-transformer for keyword / lyric embeddings ───────────────────────
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None

# ── Zero-shot NLI pipeline for emotion classification ─────────────────────────
_NLI_MODEL_NAME = "facebook/bart-large-mnli"
_nli_pipeline = None

# ── Retrieve-then-aggregate index ─────────────────────────────────────────────
_lyric_embs: np.ndarray | None = None          # (n_tracks, embed_dim)
_scaled_audio: np.ndarray | None = None        # (n_tracks, n_features)
_feature_cols: list[str] | None = None

_INDEX_ERROR_MSG = (
    "Lyric index not built. Run build_lyric_index(...) once per kernel or "
    "load a cached index via load_lyric_index(<path>) before calling resolve_keywords."
)

# Emotion labels that must match the `emotion` column values in the dataset
EMOTION_LABELS: list[str] = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

_EMOTION_ALIAS_MAP = {
    "joy": "joy",
    "anticipation": "surprise",
    "trust": "neutral",
    "disgust": "anger",
    "fear": "fear",
    "sadness": "sadness",
    "surprise": "surprise",
    "anger": "anger",
    "positive": "joy",
    "negative": "sadness",
    "neutral": "neutral",
}

# Number of nearest-neighbor tracks to aggregate for the audio target
_RETRIEVE_K = 20

# NLI confidence threshold below which a keyword is treated as a location term
_LOCATION_THRESHOLD = 0.30

# Keys used when the index is unavailable (fallback)
from data_loader import AUDIO_FEATURE_COLS as _AUDIO_FEATURE_COLS


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model=_NLI_MODEL_NAME,
        )
    return _nli_pipeline


def _embed(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ── Index API ─────────────────────────────────────────────────────────────────

def build_lyric_index(
    filtered_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    lyrics_col: str = "lyrics",
) -> None:
    """
    Pre-embed all track lyrics and cache the scaled audio feature matrix.
    Must be called once per kernel session (or `load_lyric_index` must be
    invoked) before resolve_keywords can produce meaningful audio targets.

    Parameters
    ----------
    filtered_df  : DataFrame with a lyrics column (same index as scaled_df)
    scaled_df    : z-scored audio feature DataFrame
    feature_cols : audio feature columns to use; defaults to AUDIO_FEATURE_COLS
    lyrics_col   : name of the lyrics column in filtered_df
    """
    global _lyric_embs, _scaled_audio, _feature_cols

    if feature_cols is None:
        feature_cols = _AUDIO_FEATURE_COLS

    lyrics = [
        str(t)[:512] if isinstance(t, str) else ""
        for t in filtered_df[lyrics_col].fillna("").tolist()
    ]
    batch_size = 256
    n_batches = math.ceil(len(lyrics) / batch_size)
    print(f"Building lyric index for {len(lyrics):,} tracks ({n_batches} batches) …")
    model = _get_model()
    all_embs = []
    for i in tqdm(range(n_batches), desc="Embedding lyrics", unit="batch"):
        batch = lyrics[i * batch_size : (i + 1) * batch_size]
        all_embs.append(model.encode(batch, normalize_embeddings=True, show_progress_bar=False))
    _lyric_embs = np.vstack(all_embs)
    _scaled_audio = scaled_df[feature_cols].to_numpy()    # (n_tracks, n_features)
    _feature_cols = list(feature_cols)
    print("Lyric index ready.")


def save_lyric_index(path: str | Path) -> None:
    """Persist the in-memory lyric index to disk for reuse."""
    if not is_lyric_index_ready():
        raise RuntimeError("Cannot save lyric index before it is built.")

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        lyric_embs=_lyric_embs.astype(np.float32, copy=False),
        scaled_audio=_scaled_audio.astype(np.float32, copy=False),
        feature_cols=np.array(_feature_cols, dtype="U"),
    )


def load_lyric_index(path: str | Path) -> None:
    """Load a previously cached lyric index."""
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    data = np.load(cache_path, allow_pickle=False)
    global _lyric_embs, _scaled_audio, _feature_cols
    _lyric_embs = data["lyric_embs"]
    _scaled_audio = data["scaled_audio"]
    _feature_cols = data["feature_cols"].astype(str).tolist()


def is_lyric_index_ready() -> bool:
    """Return True if the lyric index is available in memory."""
    return _lyric_embs is not None and _scaled_audio is not None and _feature_cols is not None


def _assert_index_ready() -> None:
    if not is_lyric_index_ready():
        raise RuntimeError(_INDEX_ERROR_MSG)


# ── Core resolution ───────────────────────────────────────────────────────────

def _resolve_emotion_nli(text: str) -> dict[str, float]:
    """
    Classify `text` against EMOTION_LABELS using zero-shot NLI.
    Returns a dict {label: score} where scores sum to ~1.
    """
    clf = _get_nli_pipeline()
    result = clf(text, candidate_labels=EMOTION_LABELS, multi_label=False)
    return dict(zip(result["labels"], result["scores"]))


def _normalize_emotion_label(label: str | None) -> str | None:
    if not label:
        return None
    key = label.strip().lower()
    if key in EMOTION_LABELS:
        return key
    return _EMOTION_ALIAS_MAP.get(key)


def _resolve_emotions_from_metadata(
    dominant_emotion: str | None,
    emotion_scores: dict[str, float] | None,
) -> tuple[list[str], dict[str, float]] | None:
    """
    Resolve emotions from provided metadata.

    Parameters
    ----------
    dominant_emotion : str | None
        Optional dominant emotion emitted by the Airbnb NLP pipeline.
    emotion_scores : dict[str, float] | None
        Optional normalized emotion score map from the pipeline.

    Returns
    -------
    tuple[list[str], dict[str, float]] | None
        Emotion labels sorted by inferred weight and their corresponding scores.
    """
    candidates: dict[str, float] = {}
    normalized_hint = _normalize_emotion_label(dominant_emotion)
    if normalized_hint:
        candidates[normalized_hint] = 1.0

    if emotion_scores:
        for raw_label, score in emotion_scores.items():
            normalized = _normalize_emotion_label(raw_label)
            if not normalized:
                continue
            candidates[normalized] = max(candidates.get(normalized, 0.0), float(score))

    if not candidates:
        return None

    emotions_sorted = sorted(candidates, key=lambda e: -candidates[e])
    return emotions_sorted, candidates


def _resolve_audio_retrieve(kw_embs: np.ndarray, k: int = _RETRIEVE_K) -> dict[str, float]:
    """
    Retrieve-then-aggregate logic.

    Given keyword embeddings (n_keywords, dim), find the top-k tracks by
    cosine similarity to the mean keyword embedding, then average their
    scaled audio features to form the target vector.

    Parameters
    ----------
    kw_embs : np.ndarray
        Keyword embeddings (n_keywords, dim).
    k : int, optional
        Number of nearest-neighbor tracks to aggregate (default: _RETRIEVE_K).

    Returns
    -------
    dict[str, float]
        Target z-score per audio feature or all-zeros if index missing.
    """
    _assert_index_ready()

    # Average all keyword embeddings so the query represents the overall listing vibe.
    mean_kw = kw_embs.mean(axis=0)                        # (dim,)
    mean_kw = mean_kw / (np.linalg.norm(mean_kw) + 1e-9)  # keep cosine similarity stable

    # Cosine similarity reduces to a dot product because embeddings are normalized.
    # We grab the top-k most similar lyric vectors and treat them as pseudo-neighbors
    # whose audio profiles we can average to derive the target feature vector.
    sims = _lyric_embs @ mean_kw                          # (n_tracks,)
    top_k_idx = np.argpartition(sims, -k)[-k:]
    audio_target_vec = _scaled_audio[top_k_idx].mean(axis=0)  # (n_features,)

    return {feat: float(audio_target_vec[i]) for i, feat in enumerate(_feature_cols)}


def resolve_keywords(
    keywords: list[str],
    *,
    dominant_emotion: str | None = None,
    emotion_scores: dict[str, float] | None = None,
) -> dict:
    """
    Resolve keywords to emotions, audio target, and location terms.

    Parameters
    ----------
    keywords : list[str]
        TF-IDF keywords extracted from the Airbnb listing description.
    dominant_emotion : str | None, optional
        Optional dominant emotion emitted by the Airbnb NLP pipeline.
    emotion_scores : dict[str, float] | None, optional
        Optional normalized emotion score map from the pipeline.

    Returns
    -------
    dict
        Resolved emotions, audio target, and location terms.
    """
    fallback_features = _feature_cols if _feature_cols is not None else _AUDIO_FEATURE_COLS
    if not keywords:
        return {
            "emotions": [],
            "emotion_weights": {},
            "audio_target": {f: 0.0 for f in fallback_features},
            "location_terms": [],
        }

    # ── Embed keywords (used for audio retrieval + location detection) ─────────
    kw_embs = _embed(keywords)                            # (n_keywords, dim)

    # ── Emotion — prefer provided metadata, otherwise zero-shot NLI ────────────
    metadata_emotions = _resolve_emotions_from_metadata(dominant_emotion, emotion_scores)
    if metadata_emotions is not None:
        emotions_sorted, emotion_weights = metadata_emotions
    else:
        kw_text = ", ".join(keywords)
        emotion_weights = _resolve_emotion_nli(kw_text)
        emotions_sorted = sorted(emotion_weights, key=lambda e: -emotion_weights[e])

    # ── Audio target — retrieve-then-aggregate ─────────────────────────────────
    audio_target = _resolve_audio_retrieve(kw_embs)

    # ── Location terms ─────────────────────────────────────────────────────────
    # A keyword is a location term if its individual NLI max confidence is low.
    # We classify each keyword separately and flag those below the threshold.
    location_terms: list[str] = []
    clf = _get_nli_pipeline()
    for kw in keywords:
        result = clf(kw, candidate_labels=EMOTION_LABELS, multi_label=False)
        max_conf = max(result["scores"])
        if max_conf < _LOCATION_THRESHOLD:
            location_terms.append(kw)

    return {
        "emotions": emotions_sorted,
        "emotion_weights": emotion_weights,
        "audio_target": audio_target,
        "location_terms": location_terms,
    }


def explain_resolution(keywords: list[str]) -> None:
    """Print a human-readable breakdown of how keywords were resolved."""
    resolved = resolve_keywords(keywords)
    print("=== Keyword Embedding Resolution ===")
    print(f"Input keywords   : {keywords}")
    print(f"Location terms   : {resolved['location_terms']}  ← searched in lyrics")
    print()
    print("Emotion weights (zero-shot NLI):")
    for em in resolved["emotions"]:
        score = resolved["emotion_weights"][em]
        bar = "█" * int(score * 40)
        print(f"  {em:<12} {score:.3f}  {bar}")
    print()
    index_status = "retrieve-then-aggregate" if _lyric_embs is not None else "FALLBACK (index not built)"
    print(f"Audio target vector [{index_status}]:")
    for feat, val in resolved["audio_target"].items():
        if abs(val) > 0.05:
            bar = "+" * int(abs(val) * 3) if val > 0 else "-" * int(abs(val) * 3)
            print(f"  {feat:<20} {val:+.3f}  {bar}")
