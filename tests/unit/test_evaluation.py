from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_loader import AUDIO_FEATURE_COLS
from evaluation import audio_diversity, evaluate_playlist, genre_entropy, top_k_popular_baseline


@pytest.mark.unit
def test_audio_diversity_single_row_returns_zero(toy_spotify_df: pd.DataFrame) -> None:
    one_row = toy_spotify_df.head(1)
    assert audio_diversity(one_row, AUDIO_FEATURE_COLS) == 0.0


@pytest.mark.unit
def test_audio_diversity_positive_for_distinct_rows(toy_spotify_df: pd.DataFrame) -> None:
    val = audio_diversity(toy_spotify_df.head(4), AUDIO_FEATURE_COLS)
    assert val > 0


@pytest.mark.unit
def test_genre_entropy_empty_or_missing_genre_returns_zero() -> None:
    assert genre_entropy(pd.DataFrame(), genre_col="genre") == 0.0
    assert genre_entropy(pd.DataFrame({"x": [1, 2]}), genre_col="genre") == 0.0


@pytest.mark.unit
def test_genre_entropy_higher_for_mixed_distribution() -> None:
    single = pd.DataFrame({"genre": ["pop", "pop", "pop", "pop"]})
    mixed = pd.DataFrame({"genre": ["pop", "rock", "jazz", "electronic"]})

    assert genre_entropy(mixed) > genre_entropy(single)


@pytest.mark.unit
def test_top_k_popular_baseline_sorts_and_deduplicates(toy_spotify_df: pd.DataFrame) -> None:
    out = top_k_popular_baseline(toy_spotify_df, k=3)

    assert len(out) == 3
    assert out["popularity"].is_monotonic_decreasing
    assert out.duplicated(subset=["track_name", "artist"]).sum() == 0


@pytest.mark.unit
def test_top_k_popular_baseline_raises_on_missing_column(toy_spotify_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        top_k_popular_baseline(toy_spotify_df.drop(columns=["popularity"]), k=3)


@pytest.mark.unit
def test_evaluate_playlist_returns_both_metrics(toy_spotify_df: pd.DataFrame) -> None:
    metrics = evaluate_playlist(toy_spotify_df)
    assert set(metrics.keys()) == {"audio_diversity", "genre_entropy"}
    assert np.isfinite(metrics["audio_diversity"])
    assert np.isfinite(metrics["genre_entropy"])
