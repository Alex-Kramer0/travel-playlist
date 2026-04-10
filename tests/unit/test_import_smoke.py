from __future__ import annotations

import importlib

import pandas as pd
import pytest

from data_loader import AUDIO_FEATURE_COLS


@pytest.mark.unit
def test_data_loader_module_imports() -> None:
    module = importlib.import_module("data_loader")
    assert module is not None


@pytest.mark.unit
def test_clustering_module_imports() -> None:
    module = importlib.import_module("clustering")
    assert module is not None


@pytest.mark.unit
def test_recommender_module_imports(recommender_module) -> None:
    assert recommender_module is not None


@pytest.mark.unit
def test_resolve_keywords_empty_is_safe(recommender_module) -> None:
    out = recommender_module.resolve_keywords([])

    assert out["emotions"] == []
    assert out["location_terms"] == []
    assert isinstance(out["audio_target"], dict)


@pytest.mark.unit
def test_recommend_smoke_with_mocked_resolver(monkeypatch, recommender_module, toy_spotify_df, scaled_audio_df) -> None:
    monkeypatch.setattr(
        recommender_module,
        "resolve_keywords",
        lambda keywords: {
            "emotions": ["joy"],
            "emotion_weights": {"joy": 1.0},
            "audio_target": {f: 0.1 for f in AUDIO_FEATURE_COLS},
            "location_terms": ["new york"],
        },
    )

    out = recommender_module.recommend(
        keywords=["new york"],
        df=toy_spotify_df,
        scaled_df=scaled_audio_df,
        top_n=3,
    )

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
