from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPOTIFY_CLUSTERING_DIR = PROJECT_ROOT / "spotify-clustering"
RECOMMENDATION_DIR = PROJECT_ROOT / "recommendation"

for p in [str(PROJECT_ROOT), str(SPOTIFY_CLUSTERING_DIR), str(RECOMMENDATION_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def toy_spotify_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_name": ["A", "B", "C", "D", "A", "F"],
            "artist": ["X", "Y", "Z", "W", "X", "Q"],
            "genre": ["pop", "rock", "pop", "jazz", "pop", "electronic"],
            "emotion": ["joy", "sadness", "joy", "anger", "joy", "calm"],
            "lyrics": [
                "new york skyline and city lights",
                "rain in paris makes me blue",
                "beach sunshine happy times",
                "quiet cafe in rome",
                "new york skyline and city lights",
                "late night drive downtown",
            ],
            "cluster": [0, 1, 0, 2, 0, 1],
            "danceability": [0.80, 0.30, 0.75, 0.50, 0.80, 0.60],
            "energy": [0.85, 0.20, 0.70, 0.40, 0.85, 0.55],
            "loudness": [-5.0, -12.0, -6.0, -9.0, -5.0, -8.0],
            "speechiness": [0.04, 0.06, 0.05, 0.03, 0.04, 0.045],
            "acousticness": [0.12, 0.80, 0.20, 0.60, 0.12, 0.30],
            "instrumentalness": [0.00, 0.02, 0.00, 0.10, 0.00, 0.01],
            "liveness": [0.10, 0.12, 0.15, 0.08, 0.10, 0.11],
            "valence": [0.90, 0.20, 0.85, 0.35, 0.90, 0.65],
            "tempo": [120.0, 80.0, 118.0, 95.0, 120.0, 110.0],
            "duration_s": [210, 190, 205, 240, 210, 215],
            "popularity": [85, 70, 83, 60, 84, 75],
        }
    )


@pytest.fixture
def scaled_audio_df(toy_spotify_df: pd.DataFrame) -> pd.DataFrame:
    from data_loader import AUDIO_FEATURE_COLS

    scaler = StandardScaler()
    arr = scaler.fit_transform(toy_spotify_df[AUDIO_FEATURE_COLS])
    return pd.DataFrame(arr, columns=AUDIO_FEATURE_COLS)


@pytest.fixture
def raw_spotify_like_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Artist(s)": ["Artist 1", "Artist 2", "Artist 3"],
            "song": ["Song 1", "Song 2", "Song 3"],
            "text": ["lyrics one", "lyrics two", "lyrics three"],
            "Length": ["03:30", "04:05", "bad"],
            "Loudness (db)": ["-6.5db", "-9 db", "bad"],
            "Tempo": [120.0, 95.0, 110.0],
            "Energy": [0.7, 0.5, 0.4],
            "Danceability": [0.8, 0.6, 0.5],
            "Speechiness": [0.05, 0.04, 0.06],
            "Acousticness": [0.2, 0.3, 0.4],
            "Instrumentalness": [0.0, 0.0, 0.1],
            "Liveness": [0.1, 0.12, 0.08],
            "Positiveness": [0.8, 0.4, 0.3],
            "Popularity": [75, 65, 55],
            "emotion": ["joy", "sadness", "anger"],
            "Genre": ["pop", "rock", "jazz"],
            "Good for Workout": [0, 0, 0],
        }
    )


@pytest.fixture
def tiny_random_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cols = [
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
    data = rng.normal(size=(40, len(cols)))
    return pd.DataFrame(data, columns=cols)


@pytest.fixture
def recommender_module(monkeypatch):
    stub = types.ModuleType("keyword_embedder")
    stub.resolve_keywords = lambda keywords: {
        "emotions": [],
        "emotion_weights": {},
        "audio_target": {},
        "location_terms": [],
    }
    stub.explain_resolution = lambda keywords: None

    monkeypatch.setitem(sys.modules, "keyword_embedder", stub)
    sys.modules.pop("recommender", None)
    return importlib.import_module("recommender")
