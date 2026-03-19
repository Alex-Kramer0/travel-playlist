"""
recommender.py
--------------
5-layer keyword-to-playlist recommendation pipeline.

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

Layer 5 — Artist familiarity
    If the user's Spotify top artists are provided, tracks by those artists
    get a boost.

Final score = w1*lyrics + w2*emotion + w3*audio + w4*cluster + w5*artist
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import importlib

# Import from hyphenated directory
data_loader = importlib.import_module('spotify-clustering.data_loader')
AUDIO_FEATURE_COLS = data_loader.AUDIO_FEATURE_COLS

from recommendation.keyword_embedder import resolve_keywords


_DEFAULT_WEIGHTS = {
    "lyrics": 0.30,
    "emotion": 0.20,
    "audio": 0.20,
    "cluster": 0.10,
    "artist": 0.20,
}


def _lyrics_score(
    df: pd.DataFrame,
    location_terms: list[str],
    lyrics_col: str = "lyrics",
) -> tuple[np.ndarray, list[list[str]]]:
    """
    Returns a score in [0, 1] for each track based on how many location
    terms appear in its lyrics, plus a list of which terms matched.
    """
    scores = np.zeros(len(df), dtype=float)
    matched: list[list[str]] = [[] for _ in range(len(df))]
    if not location_terms or lyrics_col not in df.columns:
        return scores, matched

    lyrics_lower = df[lyrics_col].fillna("").str.lower()
    for term in location_terms:
        pattern = re.compile(r"\b" + re.escape(term.lower()) + r"\b")
        mask = lyrics_lower.str.contains(pattern, regex=True).to_numpy(dtype=bool)
        scores += mask.astype(float)
        for idx in np.where(mask)[0]:
            matched[idx].append(term)

    return np.clip(scores, 0.0, 1.0), matched


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


def _artist_familiarity_score(
    df: pd.DataFrame,
    user_top_artists: list[str] | None,
    artist_col: str = "artist",
) -> np.ndarray:
    """
    Returns 1.0 for tracks whose artist matches any of the user's top artists,
    0.0 otherwise. Matching is case-insensitive substring to handle
    compilation entries like 'Artist1, Artist2'.
    """
    scores = np.zeros(len(df), dtype=float)
    if not user_top_artists or artist_col not in df.columns:
        return scores

    artist_lower = df[artist_col].fillna("").str.lower()
    for name in user_top_artists:
        pattern = re.compile(re.escape(name.lower()))
        matches = artist_lower.str.contains(pattern, regex=True).to_numpy(dtype=float)
        scores = np.maximum(scores, matches)

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
    explicit_location_terms: list[str] | None = None,
    user_top_artists: list[str] | None = None,
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
    explicit_location_terms : if provided, these location terms are used
                  directly for lyrics search (merged with NLI-detected ones)
    user_top_artists : if provided, tracks by these artists get a familiarity
                  boost in the scoring

    Returns
    -------
    DataFrame with columns: track_name, artist, genre, emotion, cluster,
    score, score_lyrics, score_emotion, score_audio, score_cluster,
    score_artist, + all audio feature columns
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}

    resolved = resolve_keywords(keywords)
    emotions = resolved["emotions"]
    audio_target = resolved["audio_target"]
    location_terms = resolved["location_terms"]

    # Merge explicit location terms (from NER) with NLI-detected ones
    if explicit_location_terms:
        seen = {t.lower() for t in location_terms}
        for t in explicit_location_terms:
            if t.lower() not in seen:
                location_terms.append(t)
                seen.add(t.lower())

    s_lyrics, lyrics_matched = _lyrics_score(df, location_terms, lyrics_col)
    s_emotion = _emotion_score(df, emotions, emotion_col)
    s_audio = _audio_cosine_score(scaled_df, audio_target, feature_cols)
    s_cluster = _cluster_boost_score(df, scaled_df, audio_target, cluster_col, feature_cols)
    s_artist = _artist_familiarity_score(df, user_top_artists)

    total = (
        w["lyrics"] * s_lyrics
        + w["emotion"] * s_emotion
        + w["audio"] * s_audio
        + w["cluster"] * s_cluster
        + w.get("artist", 0.0) * s_artist
    )

    result = df[["track_name", "artist", "genre", "emotion", cluster_col]].copy()
    result["score"] = total
    result["score_lyrics"] = s_lyrics
    result["score_emotion"] = s_emotion
    result["score_audio"] = s_audio
    result["score_cluster"] = s_cluster
    result["score_artist"] = s_artist
    result["matched_terms"] = lyrics_matched
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
