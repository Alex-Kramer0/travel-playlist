"""
Backend shared state module.

This module holds all heavy objects loaded at startup:
- Spotify dataset (raw, filtered, scaled)
- K-means clustering model + labels
- NRC emotion lexicon
- Lyric embedding index (cached in keyword_embedder module)

All request handlers import from this module to access shared state.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Dataset artifacts
raw_df: pd.DataFrame | None = None
filtered_df: pd.DataFrame | None = None
feature_df: pd.DataFrame | None = None
scaled_df: pd.DataFrame | None = None
scaler: StandardScaler | None = None

# Clustering artifacts
kmeans_model: KMeans | None = None
cluster_labels: list[int] | None = None

# NRC lexicon
nrc_lexicon: dict[str, set] | None = None

# Audio feature columns
AUDIO_FEATURE_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_s",
    "popularity",
]
