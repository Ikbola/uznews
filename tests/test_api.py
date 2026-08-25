"""Tests for the classification API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from uznews.api import app

client = TestClient(app)

MODEL_EXISTS = Path("results/baseline_model.joblib").exists()
needs_model = pytest.mark.skipif(
    not MODEL_EXISTS, reason="run scripts/baseline.py to train the model"
)

SPORT_TEXT = (
    "Futbol boʻyicha Oʻzbekiston terma jamoasi jahon chempionati saralash "
    "bosqichidagi navbatdagi oʻyinini oʻtkazdi. Uchrashuv Toshkentdagi "
    "stadionda boʻlib oʻtdi va jamoa gʻalaba qozondi. Murabbiy oʻyindan "
    "keyingi matbuot anjumanida futbolchilar oʻyinini yuqori baholadi."
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_rejects_short_text():
    r = client.post("/classify", json={"text": "salom"})
    assert r.status_code == 422


def test_rejects_missing_field():
    r = client.post("/classify", json={})
    assert r.status_code == 422


@needs_model
def test_classify_returns_valid_shape():
    r = client.post("/classify", json={"text": SPORT_TEXT})
    assert r.status_code == 200

    body = r.json()
    assert set(body) == {"label", "confidence", "scores"}
    assert body["label"] in body["scores"]
    assert 0.0 <= body["confidence"] <= 1.0


@needs_model
def test_scores_cover_all_labels_and_sum_to_one():
    r = client.post("/classify", json={"text": SPORT_TEXT})
    scores = r.json()["scores"]

    assert set(scores) == {"iqtisodiyot", "jahon", "jamiyat", "sport", "uzbekiston"}
    assert sum(scores.values()) == pytest.approx(1.0, abs=0.01)


@needs_model
def test_confidence_is_the_max_score():
    body = client.post("/classify", json={"text": SPORT_TEXT}).json()
    assert body["confidence"] == pytest.approx(max(body["scores"].values()))


@needs_model
def test_classifies_a_clear_sports_article_as_sport():
    body = client.post("/classify", json={"text": SPORT_TEXT}).json()
    assert body["label"] == "sport", f"got {body['label']} with scores {body['scores']}"