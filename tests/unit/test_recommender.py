from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_loader import AUDIO_FEATURE_COLS


@pytest.mark.unit
def test_lyrics_score_matches_keywords(recommender_module, toy_spotify_df: pd.DataFrame) -> None:
    scores, matched = recommender_module._lyrics_score(toy_spotify_df, ["new york", "paris"], "lyrics")

    assert scores.shape[0] == len(toy_spotify_df)
    assert scores.max() <= 1.0
    assert scores.min() >= 0.0
    assert any("new york" in m for m in matched)


@pytest.mark.unit
def test_lyrics_score_missing_column_returns_zeros(recommender_module, toy_spotify_df: pd.DataFrame) -> None:
    df = toy_spotify_df.drop(columns=["lyrics"])
    scores, matched = recommender_module._lyrics_score(df, ["new york"], "lyrics")

    assert np.allclose(scores, 0.0)
    assert len(matched) == len(df)


@pytest.mark.unit
def test_emotion_score_applies_rank_weights(recommender_module, toy_spotify_df: pd.DataFrame) -> None:
    scores = recommender_module._emotion_score(toy_spotify_df, ["joy", "sadness", "anger"], "emotion")

    joy_mask = toy_spotify_df["emotion"].str.lower() == "joy"
    sadness_mask = toy_spotify_df["emotion"].str.lower() == "sadness"

    assert scores[joy_mask].mean() > scores[sadness_mask].mean()
    assert scores.max() <= 1.0


@pytest.mark.unit
def test_audio_cosine_score_zero_target_returns_half(recommender_module, scaled_audio_df: pd.DataFrame) -> None:
    zero_target = {f: 0.0 for f in AUDIO_FEATURE_COLS}
    scores = recommender_module._audio_cosine_score(scaled_audio_df, zero_target, AUDIO_FEATURE_COLS)

    assert np.allclose(scores, 0.5)


@pytest.mark.unit
def test_cluster_boost_marks_nearest_cluster(
    recommender_module,
    toy_spotify_df: pd.DataFrame,
    scaled_audio_df: pd.DataFrame,
) -> None:
    means = (
        scaled_audio_df.assign(cluster=toy_spotify_df["cluster"].values)
        .groupby("cluster")[AUDIO_FEATURE_COLS]
        .mean()
    )
    target_cluster = int(means.index[0])
    audio_target = means.loc[target_cluster].to_dict()

    scores = recommender_module._cluster_boost_score(
        toy_spotify_df,
        scaled_audio_df,
        audio_target,
        cluster_col="cluster",
        feature_cols=AUDIO_FEATURE_COLS,
    )

    boosted_clusters = toy_spotify_df.loc[scores == 1.0, "cluster"].unique().tolist()
    assert boosted_clusters == [target_cluster]


@pytest.mark.unit
def test_recommend_returns_expected_columns_and_top_n(
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
            "audio_target": {f: 0.1 for f in AUDIO_FEATURE_COLS},
            "location_terms": ["new york"],
        },
    )

    result = recommender_module.recommend(
        keywords=["new york", "city"],
        df=toy_spotify_df,
        scaled_df=scaled_audio_df,
        top_n=4,
        feature_cols=AUDIO_FEATURE_COLS,
    )

    expected = {
        "track_name",
        "artist",
        "genre",
        "emotion",
        "cluster",
        "score",
        "score_lyrics",
        "score_emotion",
        "score_audio",
        "score_cluster",
        "matched_keywords",
    }
    assert len(result) == 4
    assert expected.issubset(result.columns)


@pytest.mark.unit
def test_recommend_deduplicates_track_artist_pairs(
    recommender_module,
    monkeypatch,
    toy_spotify_df: pd.DataFrame,
    scaled_audio_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        recommender_module,
        "resolve_keywords",
        lambda keywords: {
            "emotions": [],
            "emotion_weights": {},
            "audio_target": {f: 0.0 for f in AUDIO_FEATURE_COLS},
            "location_terms": [],
        },
    )

    result = recommender_module.recommend(
        keywords=["anything"],
        df=toy_spotify_df,
        scaled_df=scaled_audio_df,
        top_n=10,
        deduplicate=True,
    )

    assert result.duplicated(subset=["track_name", "artist"]).sum() == 0
