from __future__ import annotations

import pandas as pd
import pytest


class _IdentityLemmatizer:
    def lemmatize(self, token: str) -> str:
        return token


def _patch_lightweight_nlp(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "word_tokenize", lambda text: text.split())

    def fake_pos(tokens):
        tags = {
            "cozy": "JJ",
            "view": "NN",
            "bright": "JJ",
            "room": "NN",
            "calm": "JJ",
            "space": "NN",
        }
        return [(t, tags.get(t, "NN")) for t in tokens]

    monkeypatch.setattr(module, "pos_tag", fake_pos)
    monkeypatch.setattr(module, "lemmatizer", _IdentityLemmatizer())


@pytest.mark.integration
def test_analyze_listing_requires_exactly_one_input(airbnb_nlp_module) -> None:
    with pytest.raises(ValueError):
        airbnb_nlp_module.analyze_listing("dataset.csv", "nrc.txt")

    with pytest.raises(ValueError):
        airbnb_nlp_module.analyze_listing("dataset.csv", "nrc.txt", url="u", description="d")


@pytest.mark.integration
def test_analyze_listing_description_returns_keyword_and_emotion_json(
    tmp_path,
    monkeypatch,
    airbnb_nlp_module,
) -> None:
    _patch_lightweight_nlp(monkeypatch, airbnb_nlp_module)

    nrc_path = tmp_path / "nrc_sample.txt"
    nrc_path.write_text(
        "cozy\tjoy\t1\n"
        "bright\tjoy\t1\n"
        "room\ttrust\t1\n"
        "cozy\tpositive\t1\n",
        encoding="utf-8",
    )

    listing_data = {
        "id": 123,
        "listing_url": "http://example.com/listing",
        "name": "Sample Stay",
        "neighbourhood_cleansed": "Downtown",
    }

    keyword_json, emotion_json = airbnb_nlp_module.analyze_listing_description(
        description="Cozy view and bright room",
        nrc_path=str(nrc_path),
        listing_data=listing_data,
    )

    assert keyword_json["id"] == 123
    assert isinstance(keyword_json["keywords"], list)
    assert "cozy view" in keyword_json["keywords"]
    assert emotion_json["dominant_emotion"] in {"joy", "trust"}
    assert "joy" in emotion_json["emotion_scores"]


@pytest.mark.integration
def test_analyze_listing_url_flow_uses_dataset_lookup(
    tmp_path,
    monkeypatch,
    airbnb_nlp_module,
) -> None:
    _patch_lightweight_nlp(monkeypatch, airbnb_nlp_module)

    dataset_path = tmp_path / "listings.csv"
    pd.DataFrame(
        [
            {
                "id": 1,
                "listing_url": "http://example.com/a",
                "name": "A",
                "description": "Cozy view bright room",
                "neighbourhood_cleansed": "Center",
            }
        ]
    ).to_csv(dataset_path, index=False)

    nrc_path = tmp_path / "nrc_sample.txt"
    nrc_path.write_text("cozy\tjoy\t1\ncozy\tpositive\t1\n", encoding="utf-8")

    keyword_json, emotion_json = airbnb_nlp_module.analyze_listing(
        dataset_path=str(dataset_path),
        nrc_path=str(nrc_path),
        url="http://example.com/a",
    )

    assert keyword_json["listing_url"] == "http://example.com/a"
    assert isinstance(keyword_json["keywords"], list)
    assert "positive_norm" in emotion_json
