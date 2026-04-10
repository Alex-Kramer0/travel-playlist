from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clustering import build_cluster_profile, evaluate_kmeans, fit_kmeans


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
