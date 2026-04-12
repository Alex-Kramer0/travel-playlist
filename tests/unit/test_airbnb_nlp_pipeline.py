from __future__ import annotations

import pytest


@pytest.mark.unit
def test_normalize_text_strips_html_punct_numbers_and_city(airbnb_nlp_module) -> None:
    text = "<b>Fort Lauderdale</b> Cozy!!! 2-bedroom with café vibes."
    out = airbnb_nlp_module.normalize_text(text)

    assert "fort lauderdale" not in out
    assert "<b>" not in out
    assert "2" not in out
    assert "cozy" in out


@pytest.mark.unit
def test_keep_vibe_phrase_filters_blocked_phrases(airbnb_nlp_module) -> None:
    assert airbnb_nlp_module.keep_vibe_phrase("hard_rock") is False
    assert airbnb_nlp_module.keep_vibe_phrase("great_pool") is False
    assert airbnb_nlp_module.keep_vibe_phrase("cozy_view") is True


@pytest.mark.unit
def test_load_nrc_lexicon_parses_valid_entries(tmp_path, airbnb_nlp_module) -> None:
    nrc_path = tmp_path / "nrc_sample.txt"
    nrc_path.write_text(
        "# comment\n"
        "cozy\tjoy\t1\n"
        "storm\tfear\t1\n"
        "bad\tnegative\t1\n"
        "fake\tjoy\t0\n",
        encoding="utf-8",
    )

    lex = airbnb_nlp_module.load_nrc_lexicon(str(nrc_path))

    assert "cozy" in lex["joy"]
    assert "storm" in lex["fear"]
    assert "bad" in lex["negative"]
    assert "fake" not in lex["joy"]


@pytest.mark.unit
def test_score_nrc_empty_tokens_returns_safe_defaults(airbnb_nlp_module) -> None:
    out = airbnb_nlp_module.score_nrc([], {"joy": {"happy"}})

    assert out["n_tokens"] == 0
    assert out["dominant_emotion"] is None
    assert out["pos_neg_ratio"] is None
    assert out["joy_norm"] == 0.0


@pytest.mark.unit
def test_score_nrc_counts_and_dominant_emotion(airbnb_nlp_module) -> None:
    cat_to_words = {
        "joy": {"cozy", "bright"},
        "fear": {"storm"},
        "positive": {"cozy", "bright"},
        "negative": {"storm"},
    }
    out = airbnb_nlp_module.score_nrc(["cozy", "bright", "storm"], cat_to_words)

    assert out["joy"] == 2
    assert out["fear"] == 1
    assert out["dominant_emotion"] == "joy"
    assert out["n_tokens"] == 3
