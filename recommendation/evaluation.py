"""
evaluation.py
-------------
Playlist-level evaluation metrics for comparing a generated playlist
against a popularity-based baseline.

Metrics
-------
1. Intra-playlist audio diversity
   Mean pairwise cosine *distance* (1 − similarity) across all track pairs'
   audio feature vectors. Higher → more sonically diverse.

2. Genre entropy
   Shannon entropy H = −Σ p_g log p_g over the genre distribution.
   Higher → broader genre coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import AUDIO_FEATURE_COLS


def audio_diversity(
    df: pd.DataFrame,
    audio_cols: list[str] = AUDIO_FEATURE_COLS,
) -> float:
    """Mean pairwise cosine distance (1 − sim) over audio feature vectors."""
    cols = [c for c in audio_cols if c in df.columns]
    if len(cols) == 0 or len(df) < 2:
        return 0.0

    mat = df[cols].to_numpy(dtype=float)
    sim_matrix = cosine_similarity(mat)
    n = len(sim_matrix)
    # extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    pairwise_dists = 1.0 - sim_matrix[triu_idx]
    return float(np.mean(pairwise_dists))


def genre_entropy(
    df: pd.DataFrame,
    genre_col: str = "genre",
) -> float:
    """Shannon entropy over genre distribution in the playlist."""
    if genre_col not in df.columns or len(df) == 0:
        return 0.0

    counts = df[genre_col].fillna("unknown").value_counts()
    probs = counts / counts.sum()
    # H = -Σ p log p  (use natural log; 0·log0 = 0)
    entropy = -float((probs * np.log(probs + 1e-12)).sum())
    return entropy


def evaluate_playlist(
    df: pd.DataFrame,
    audio_cols: list[str] = AUDIO_FEATURE_COLS,
    genre_col: str = "genre",
) -> dict[str, float]:
    """Return a dict with both evaluation metrics for a playlist DataFrame."""
    return {
        "audio_diversity": audio_diversity(df, audio_cols),
        "genre_entropy": genre_entropy(df, genre_col),
    }


def top_k_popular_baseline(
    full_df: pd.DataFrame,
    k: int,
    popularity_col: str = "popularity",
) -> pd.DataFrame:
    """Return the top-k tracks by popularity as a naive baseline playlist."""
    if popularity_col not in full_df.columns:
        raise ValueError(f"Column '{popularity_col}' not found in DataFrame")
    return (
        full_df
        .sort_values(popularity_col, ascending=False)
        .drop_duplicates(subset=["track_name", "artist"])
        .head(k)
        .reset_index(drop=True)
    )
