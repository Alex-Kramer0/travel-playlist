from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    AUDIO_FEATURE_COLS,
    CLUSTER_FEATURE_COLS,
    _parse_length,
    _parse_loudness,
    load_spotify,
    pca_reduce,
    quantile_transform,
    remove_outliers,
    scale_features,
    select_features,
)


@pytest.mark.unit
def test_parse_loudness_handles_db_and_invalid() -> None:
    s = pd.Series(["-6.85db", "-10 db", "bad"])
    out = _parse_loudness(s)
    assert out.iloc[0] == pytest.approx(-6.85)
    assert out.iloc[1] == pytest.approx(-10.0)
    assert np.isnan(out.iloc[2])


@pytest.mark.unit
def test_parse_length_handles_mm_ss_and_invalid() -> None:
    s = pd.Series(["03:45", "0:59", "bad"])
    out = _parse_length(s)
    assert out.iloc[0] == 225
    assert out.iloc[1] == 59
    assert np.isnan(out.iloc[2])


@pytest.mark.unit
def test_load_spotify_renames_and_drops_prefixed_columns(
    tmp_path, raw_spotify_like_df: pd.DataFrame
) -> None:
    path = tmp_path / "raw.csv"
    raw_spotify_like_df.to_csv(path, index=False)

    df = load_spotify(str(path))

    assert "artist" in df.columns
    assert "track_name" in df.columns
    assert "lyrics" in df.columns
    assert "duration_s" in df.columns
    assert "Good for Workout" not in df.columns


@pytest.mark.unit
def test_select_features_raises_on_missing_feature(toy_spotify_df: pd.DataFrame) -> None:
    bad_df = toy_spotify_df.drop(columns=["popularity"])
    with pytest.raises(ValueError):
        select_features(bad_df, AUDIO_FEATURE_COLS)


@pytest.mark.unit
def test_select_features_drops_nan_rows(toy_spotify_df: pd.DataFrame) -> None:
    df = toy_spotify_df.copy()
    df.loc[0, "energy"] = np.nan

    filtered_df, feature_df = select_features(df, AUDIO_FEATURE_COLS)

    assert len(filtered_df) == len(df) - 1
    assert feature_df["energy"].isna().sum() == 0


@pytest.mark.unit
def test_remove_outliers_removes_extreme_row(toy_spotify_df: pd.DataFrame) -> None:
    filtered_df, feature_df = select_features(toy_spotify_df, AUDIO_FEATURE_COLS)
    feature_df = feature_df.copy()
    extreme = {c: feature_df[c].median() for c in AUDIO_FEATURE_COLS}
    extreme["tempo"] = 100_000.0
    feature_df = pd.concat([feature_df, pd.DataFrame([extreme])], ignore_index=True)
    filtered_df = pd.concat([filtered_df, filtered_df.head(1)], ignore_index=True)

    clean_df, clean_features = remove_outliers(filtered_df, feature_df, AUDIO_FEATURE_COLS, z_threshold=2.0)

    assert len(clean_df) <= len(filtered_df) - 1
    assert len(clean_features) <= len(feature_df) - 1


@pytest.mark.unit
def test_scale_features_standardizes_columns(toy_spotify_df: pd.DataFrame) -> None:
    _, feature_df = select_features(toy_spotify_df, AUDIO_FEATURE_COLS)
    scaled_df, _ = scale_features(feature_df, AUDIO_FEATURE_COLS)

    means = scaled_df.mean(axis=0).abs()
    stds = scaled_df.std(axis=0, ddof=0)

    assert (means < 1e-7).all()
    assert np.allclose(stds.values, np.ones(len(AUDIO_FEATURE_COLS)), atol=1e-6)


@pytest.mark.unit
def test_quantile_transform_preserves_shape(tiny_random_features: pd.DataFrame) -> None:
    qt_df, _ = quantile_transform(tiny_random_features, CLUSTER_FEATURE_COLS)
    assert qt_df.shape == (len(tiny_random_features), len(CLUSTER_FEATURE_COLS))
    assert list(qt_df.columns) == CLUSTER_FEATURE_COLS


@pytest.mark.unit
def test_pca_reduce_returns_expected_components(tiny_random_features: pd.DataFrame) -> None:
    qt_df, _ = quantile_transform(tiny_random_features, CLUSTER_FEATURE_COLS)
    pca_df, _ = pca_reduce(qt_df, n_components=2)

    assert list(pca_df.columns) == ["PC1", "PC2"]
    assert len(pca_df) == len(tiny_random_features)
