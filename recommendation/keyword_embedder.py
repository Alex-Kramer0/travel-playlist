"""
keyword_embedder.py
-------------------
Embedding-based mapping from Airbnb TF-IDF keywords to:
  - target emotion label(s)   via zero-shot NLI  (facebook/bart-large-mnli)
  - target audio feature vector via retrieve-then-aggregate over the track catalog

Approach
--------
1. Embed each input keyword with sentence-transformers (all-MiniLM-L6-v2).

2. EMOTION — Zero-shot NLI:
   Concatenate keywords into a short passage and classify against the emotion
   labels using a zero-shot NLI pipeline (facebook/bart-large-mnli).
   No hand-crafted anchor phrases needed.

3. AUDIO — Retrieve-then-aggregate:
   Find the top-k tracks in the catalog whose lyric embeddings are most similar
   to the mean keyword embedding (cosine similarity). Average their z-scored
   audio features to form the target vector. Fully data-driven — no anchor
   phrases needed.

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

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
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

# Emotion labels that must match the `emotion` column values in the dataset
EMOTION_LABELS: list[str] = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

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
    Must be called once after loading the dataset before resolve_keywords
    can produce meaningful audio targets.

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

    lyrics = filtered_df[lyrics_col].fillna("").tolist()
    print(f"Building lyric index for {len(lyrics)} tracks …")
    _lyric_embs = _embed(lyrics)                          # (n_tracks, dim)
    _scaled_audio = scaled_df[feature_cols].to_numpy()    # (n_tracks, n_features)
    _feature_cols = list(feature_cols)
    print("Lyric index ready.")


# ── Core resolution ───────────────────────────────────────────────────────────

def _resolve_emotion_nli(text: str) -> dict[str, float]:
    """
    Classify `text` against EMOTION_LABELS using zero-shot NLI.
    Returns a dict {label: score} where scores sum to ~1.
    """
    clf = _get_nli_pipeline()
    result = clf(text, candidate_labels=EMOTION_LABELS, multi_label=False)
    return dict(zip(result["labels"], result["scores"]))


def _resolve_audio_retrieve(kw_embs: np.ndarray, k: int = _RETRIEVE_K) -> dict[str, float]:
    """
    Given keyword embeddings (n_keywords, dim), find the top-k tracks by
    cosine similarity to the mean keyword embedding, then average their
    scaled audio features to form the target vector.

    Returns a dict {feature: z_score_target} or all-zeros if index missing.
    """
    if _lyric_embs is None or _scaled_audio is None or _feature_cols is None:
        return {f: 0.0 for f in _AUDIO_FEATURE_COLS}

    mean_kw = kw_embs.mean(axis=0)                        # (dim,)
    mean_kw = mean_kw / (np.linalg.norm(mean_kw) + 1e-9)

    sims = _lyric_embs @ mean_kw                          # (n_tracks,)
    top_k_idx = np.argpartition(sims, -k)[-k:]
    audio_target_vec = _scaled_audio[top_k_idx].mean(axis=0)  # (n_features,)

    return {feat: float(audio_target_vec[i]) for i, feat in enumerate(_feature_cols)}


def resolve_keywords(keywords: list[str]) -> dict:
    """
    Given a list of Airbnb TF-IDF keywords, resolve:
      - emotion weights  via zero-shot NLI on the concatenated keyword string
      - audio target     via retrieve-then-aggregate over the lyric index
      - location terms   via low NLI confidence (no strong emotion signal)

    Parameters
    ----------
    keywords : list of keyword strings

    Returns
    -------
    dict with keys:
        "emotions"       : list[str]  — emotion labels sorted by inferred weight
        "emotion_weights": dict[str, float]  — NLI confidence scores per emotion
        "audio_target"   : dict[str, float]  — target z-score per audio feature
        "location_terms" : list[str]  — keywords passed to the lyrics-search layer
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

    # ── Emotion — zero-shot NLI ────────────────────────────────────────────────
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
