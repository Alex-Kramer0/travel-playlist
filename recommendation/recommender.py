"""
recommender.py
--------------
4-layer keyword-to-playlist recommendation pipeline.

Layer 1 — Lyrics keyword match
    Location/place terms (e.g. "new york", "paris") are searched directly
    in the lyrics column. Tracks that mention the term get a strong boost.

Layer 2 — Emotion match
    Keywords are embedded with sentence-transformers and compared to emotion
    anchor embeddings. Tracks whose emotion column matches the top inferred
    emotion(s) are boosted, weighted by cosine similarity score.

Layer 3 — Audio cosine similarity
    Keywords resolve to a target audio feature vector. Each track's scaled
    audio features are compared to that vector via cosine similarity.

Layer 4 — Cluster boost
    The K-means cluster whose centroid is closest (L2) to the target vector
    receives a score multiplier. Tracks in that cluster get a boost.

Final score = w1*lyrics + w2*emotion + w3*audio_cosine + w4*cluster_boost
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import AUDIO_FEATURE_COLS
from keyword_embedder import resolve_keywords


_DEFAULT_WEIGHTS = {
    "lyrics": 0.35,
    "emotion": 0.25,
    "audio": 0.25,
    "cluster": 0.15,
}


def _lyrics_score(
    df: pd.DataFrame,
    location_terms: list[str],
    lyrics_col: str = "lyrics",
) -> np.ndarray:
    """
    Returns a score in [0, 1] for each track based on how many location
    terms appear in its lyrics. Multiple matches accumulate additively,
    then the result is clipped to 1.
    """
    scores = np.zeros(len(df), dtype=float)
    if not location_terms or lyrics_col not in df.columns:
        return scores

    lyrics_lower = df[lyrics_col].fillna("").str.lower()
    for term in location_terms:
        pattern = re.compile(r"\b" + re.escape(term.lower()) + r"\b")
        matches = lyrics_lower.str.contains(pattern, regex=True).to_numpy(dtype=float)
        scores += matches

    return np.clip(scores, 0.0, 1.0)


def _emotion_score(
    df: pd.DataFrame,
    emotions: list[str],
    emotion_col: str = "emotion",
) -> np.ndarray:
    """
    Returns a score in [0, 1] for each track based on emotion match.
    The top-ranked emotion scores 1.0, second 0.5, rest 0.25.
    """
    scores = np.zeros(len(df), dtype=float)
    if not emotions or emotion_col not in df.columns:
        return scores

    emotion_lower = df[emotion_col].fillna("").str.lower()
    weights = [1.0, 0.5] + [0.25] * max(0, len(emotions) - 2)
    for em, w in zip(emotions, weights):
        scores += (emotion_lower == em.lower()).to_numpy(dtype=float) * w

    return np.clip(scores, 0.0, 1.0)


def _audio_cosine_score(
    scaled_df: pd.DataFrame,
    audio_target: dict[str, float],
    feature_cols: list[str] = AUDIO_FEATURE_COLS,
) -> np.ndarray:
    """
    Returns cosine similarity in [-1, 1] between each track's scaled
    feature vector and the keyword-derived target vector, then rescaled
    to [0, 1].
    """
    target_vec = np.array([audio_target.get(f, 0.0) for f in feature_cols]).reshape(1, -1)
    target_norm = np.linalg.norm(target_vec)
    if target_norm < 1e-9:
        return np.full(len(scaled_df), 0.5)

    track_matrix = scaled_df[feature_cols].to_numpy()
    sims = cosine_similarity(track_matrix, target_vec).flatten()
    return (sims + 1.0) / 2.0


def _cluster_boost_score(
    df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    audio_target: dict[str, float],
    cluster_col: str = "cluster",
    feature_cols: list[str] = AUDIO_FEATURE_COLS,
) -> np.ndarray:
    """
    Finds the cluster whose centroid is closest (L2) to the audio target
    vector and returns 1.0 for tracks in that cluster, 0.0 otherwise.
    """
    scores = np.zeros(len(df), dtype=float)
    if cluster_col not in df.columns:
        return scores

    target_vec = np.array([audio_target.get(f, 0.0) for f in feature_cols])
    cluster_ids = df[cluster_col].unique()

    best_cluster = None
    best_dist = float("inf")
    for cid in cluster_ids:
        if cid == -1:
            continue
        mask = df[cluster_col] == cid
        centroid = scaled_df.loc[mask, feature_cols].mean().to_numpy()
        dist = np.linalg.norm(centroid - target_vec)
        if dist < best_dist:
            best_dist = dist
            best_cluster = cid

    if best_cluster is not None:
        scores[df[cluster_col] == best_cluster] = 1.0

    return scores


def recommend(
    keywords: list[str],
    df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    top_n: int = 20,
    weights: dict[str, float] | None = None,
    cluster_col: str = "cluster",
    lyrics_col: str = "lyrics",
    emotion_col: str = "emotion",
    feature_cols: list[str] = AUDIO_FEATURE_COLS,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """
    Recommend tracks given a list of Airbnb-style keywords.

    Parameters
    ----------
    keywords    : list of keyword strings from Airbnb TF-IDF extraction
    df          : filtered_df with metadata + cluster labels
    scaled_df   : z-scored audio feature DataFrame (same index as df)
    top_n       : number of tracks to return
    weights     : override default layer weights (must sum to ~1)
    cluster_col : column in df holding K-means cluster labels
    lyrics_col  : column in df holding song lyrics
    emotion_col : column in df holding emotion labels
    feature_cols: audio feature columns to use for scoring
    deduplicate : if True, keep only the highest-scoring version of each
                  (track_name, artist) pair

    Returns
    -------
    DataFrame with columns: track_name, artist, genre, emotion, cluster,
    score, score_lyrics, score_emotion, score_audio, score_cluster,
    + all audio feature columns
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}

    resolved = resolve_keywords(keywords)
    emotions = resolved["emotions"]
    audio_target = resolved["audio_target"]
    location_terms = resolved["location_terms"]

    s_lyrics = _lyrics_score(df, location_terms, lyrics_col)
    s_emotion = _emotion_score(df, emotions, emotion_col)
    s_audio = _audio_cosine_score(scaled_df, audio_target, feature_cols)
    s_cluster = _cluster_boost_score(df, scaled_df, audio_target, cluster_col, feature_cols)

    total = (
        w["lyrics"] * s_lyrics
        + w["emotion"] * s_emotion
        + w["audio"] * s_audio
        + w["cluster"] * s_cluster
    )

    result = df[["track_name", "artist", "genre", "emotion", cluster_col]].copy()
    result["score"] = total
    result["score_lyrics"] = s_lyrics
    result["score_emotion"] = s_emotion
    result["score_audio"] = s_audio
    result["score_cluster"] = s_cluster
    for col in feature_cols:
        if col in df.columns:
            result[col] = df[col].values

    result = result.sort_values("score", ascending=False)

    if deduplicate:
        result = result.drop_duplicates(subset=["track_name", "artist"])

    return result.head(top_n).reset_index(drop=True)


def explain_recommendation(keywords: list[str]) -> None:
    """Print a human-readable breakdown of how keywords were resolved via embeddings."""
    from keyword_embedder import explain_resolution
    explain_resolution(keywords)
