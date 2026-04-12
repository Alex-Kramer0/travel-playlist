from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clustering import (
    build_cluster_profile,
    compute_kdistance,
    evaluate_kmeans,
    fit_dbscan_full,
    fit_dbscan_with_adjustment,
    fit_kmeans,
    select_best_dbscan_params,
)


@pytest.mark.unit
def test_fit_kmeans_returns_label_for_each_row(tiny_random_features: pd.DataFrame) -> None:
    model, labels = fit_kmeans(tiny_random_features, k=3, random_state=42)

    assert model.n_clusters == 3
    assert len(labels) == len(tiny_random_features)


@pytest.mark.unit
def test_fit_kmeans_labels_in_valid_range(tiny_random_features: pd.DataFrame) -> None:
    _, labels = fit_kmeans(tiny_random_features, k=4, random_state=42)

    assert labels.min() >= 0
    assert labels.max() <= 3


@pytest.mark.unit
def test_evaluate_kmeans_returns_row_per_k(tiny_random_features: pd.DataFrame) -> None:
    result = evaluate_kmeans(tiny_random_features, k_values=range(2, 6), random_state=42)

    assert len(result) == 4
    assert set(["k", "inertia", "silhouette"]).issubset(result.columns)
    assert list(result["k"]) == [2, 3, 4, 5]


@pytest.mark.unit
def test_build_cluster_profile_shape(tiny_random_features: pd.DataFrame) -> None:
    _, labels = fit_kmeans(tiny_random_features, k=3, random_state=42)
    profile = build_cluster_profile(tiny_random_features, labels)

    assert profile.shape[0] == tiny_random_features.shape[1]
    assert profile.shape[1] == 3
    assert np.all(np.isfinite(profile.to_numpy()))


@pytest.mark.unit
def test_compute_kdistance_returns_sorted_distances_and_eps(
    tiny_random_features: pd.DataFrame,
) -> None:
    pca_df = tiny_random_features.iloc[:, :2].copy()
    kth_distances, knee_eps = compute_kdistance(pca_df, k=3, sample_size=20, knee_percentile=60)

    assert len(kth_distances) == min(20, len(pca_df))
    assert np.all(np.diff(kth_distances) >= 0)
    assert knee_eps >= 0


@pytest.mark.unit
def test_select_best_dbscan_params_prefers_target_and_noise_constrained() -> None:
    results = pd.DataFrame(
        [
            {"eps": 0.10, "min_samples": 8, "clusters": 5, "noise_ratio": 0.20, "davies_bouldin": 0.90},
            {"eps": 0.20, "min_samples": 8, "clusters": 5, "noise_ratio": 0.10, "davies_bouldin": 0.70},
            {"eps": 0.30, "min_samples": 12, "clusters": 7, "noise_ratio": 0.05, "davies_bouldin": 0.50},
        ]
    )

    best_eps, best_min_samples = select_best_dbscan_params(results)

    assert best_eps == 0.20
    assert best_min_samples == 8


@pytest.mark.unit
def test_select_best_dbscan_params_falls_back_when_noise_threshold_not_met() -> None:
    results = pd.DataFrame(
        [
            {"eps": 0.10, "min_samples": 8, "clusters": 5, "noise_ratio": 0.80, "davies_bouldin": 0.60},
            {"eps": 0.20, "min_samples": 10, "clusters": 5, "noise_ratio": 0.70, "davies_bouldin": 0.50},
            {"eps": 0.30, "min_samples": 12, "clusters": 8, "noise_ratio": 0.10, "davies_bouldin": 0.40},
        ]
    )

    best_eps, best_min_samples = select_best_dbscan_params(results)

    assert best_eps == 0.20
    assert best_min_samples == 10


@pytest.mark.unit
def test_fit_dbscan_full_returns_expected_keys(tiny_random_features: pd.DataFrame) -> None:
    pca_df = tiny_random_features.iloc[:, :2].copy()
    out = fit_dbscan_full(pca_df, eps=0.8, min_samples=3)

    expected_keys = {"eps", "min_samples", "labels", "noise_mask", "cluster_count", "noise_ratio"}
    assert expected_keys.issubset(out.keys())
    assert len(out["labels"]) == len(pca_df)
    assert len(out["noise_mask"]) == len(pca_df)
    assert 0.0 <= out["noise_ratio"] <= 1.0


@pytest.mark.unit
def test_fit_dbscan_with_adjustment_stops_when_in_target_band(tiny_random_features: pd.DataFrame) -> None:
    pca_df = tiny_random_features.iloc[:, :2].copy()
    out = fit_dbscan_with_adjustment(
        pca_df,
        initial_eps=0.7,
        min_samples=3,
        target_cluster_min=1,
        target_cluster_max=10,
        max_adjustments=3,
    )

    assert 1 <= out["cluster_count"] <= 10


@pytest.mark.unit
def test_fit_dbscan_with_adjustment_uses_closest_attempt_fallback(monkeypatch) -> None:
    pca_df = pd.DataFrame({"PC1": [0.0, 1.0], "PC2": [0.0, 1.0]})
    scripted = iter(
        [
            {"eps": 1.0, "min_samples": 5, "labels": np.array([0, 1]), "noise_mask": np.array([False, False]), "cluster_count": 1, "noise_ratio": 0.30},
            {"eps": 0.8, "min_samples": 5, "labels": np.array([0, 1]), "noise_mask": np.array([False, False]), "cluster_count": 10, "noise_ratio": 0.20},
            {"eps": 0.64, "min_samples": 5, "labels": np.array([0, 1]), "noise_mask": np.array([False, False]), "cluster_count": 8, "noise_ratio": 0.10},
        ]
    )

    def fake_fit_dbscan_full(*_args, **_kwargs):
        return next(scripted)

    monkeypatch.setattr("clustering.fit_dbscan_full", fake_fit_dbscan_full)
    out = fit_dbscan_with_adjustment(
        pca_df,
        initial_eps=1.0,
        min_samples=5,
        target_cluster_min=4,
        target_cluster_max=6,
        max_adjustments=2,
    )

    assert out["cluster_count"] == 8
    assert out["noise_ratio"] == 0.10
