from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clustering import fit_kmeans
from data_loader import (
    AUDIO_FEATURE_COLS,
    CLUSTER_FEATURE_COLS,
    load_spotify,
    pca_reduce,
    quantile_transform,
    remove_outliers,
    scale_features,
    select_features,
)
from evaluation import evaluate_playlist, top_k_popular_baseline


@pytest.mark.integration
def test_full_preprocessing_chain_from_csv(tmp_path, raw_spotify_like_df: pd.DataFrame) -> None:
    csv_path = tmp_path / "spotify_small.csv"
    raw_spotify_like_df.to_csv(csv_path, index=False)

    raw = load_spotify(str(csv_path))
    filt, feat = select_features(raw, AUDIO_FEATURE_COLS)
    clean, clean_feat = remove_outliers(filt, feat, AUDIO_FEATURE_COLS, z_threshold=4.0)
    scaled, _ = scale_features(clean_feat, AUDIO_FEATURE_COLS)

    assert len(raw) == len(raw_spotify_like_df)
    assert len(clean) == len(clean_feat)
    assert len(clean) > 0
    assert scaled.shape[1] == len(AUDIO_FEATURE_COLS)


@pytest.mark.integration
def test_clustering_chain_returns_label_for_each_track(toy_spotify_df: pd.DataFrame) -> None:
    _, feat = select_features(toy_spotify_df, AUDIO_FEATURE_COLS)
    qt_df, _ = quantile_transform(feat, CLUSTER_FEATURE_COLS)
    pca_df, _ = pca_reduce(qt_df, n_components=2)
    _, labels = fit_kmeans(pca_df, k=3, random_state=42)

    assert len(labels) == len(toy_spotify_df)


@pytest.mark.integration
def test_recommendation_and_evaluation_end_to_end(
    recommender_module,
    monkeypatch,
    toy_spotify_df: pd.DataFrame,
    scaled_audio_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        recommender_module,
        "resolve_keywords",
        lambda keywords: {
            "emotions": ["joy", "sadness"],
            "emotion_weights": {"joy": 0.7, "sadness": 0.3},
            "audio_target": {f: 0.0 for f in AUDIO_FEATURE_COLS},
            "location_terms": ["new york"],
        },
    )

    playlist = recommender_module.recommend(
        keywords=["new york", "cozy"],
        df=toy_spotify_df,
        scaled_df=scaled_audio_df,
        top_n=4,
    )
    baseline = top_k_popular_baseline(toy_spotify_df, k=4)

    ours = evaluate_playlist(playlist)
    base = evaluate_playlist(baseline)

    assert len(playlist) == 4
    assert set(ours.keys()) == {"audio_diversity", "genre_entropy"}
    assert set(base.keys()) == {"audio_diversity", "genre_entropy"}


@pytest.mark.integration
def test_kmeans_reproducibility_with_fixed_seed(toy_spotify_df: pd.DataFrame) -> None:
    _, feat = select_features(toy_spotify_df, AUDIO_FEATURE_COLS)
    qt_df, _ = quantile_transform(feat, CLUSTER_FEATURE_COLS)
    pca_df, _ = pca_reduce(qt_df, n_components=2)

    _, labels_a = fit_kmeans(pca_df, k=3, random_state=42)
    _, labels_b = fit_kmeans(pca_df, k=3, random_state=42)

    assert np.array_equal(labels_a, labels_b)
