"""
test_api.py
-----------
Tests unitaires des 3 endpoints de l'API : /health, /predict, /explain.

Lancement :
    pytest -v
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body
    assert body["model_loaded"] is True


def test_predict_positive_text(client):
    response = client.post("/predict", json={"text": "I absolutely love this, amazing quality!"})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] in ("positive", "negative")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_negative_text(client):
    response = client.post("/predict", json={"text": "This is terrible and broke immediately."})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] in ("positive", "negative")


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_explain_returns_top_words(client):
    response = client.post(
        "/explain", json={"text": "Amazing quality and fast shipping", "top_n": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] in ("positive", "negative")
    assert isinstance(body["top_words"], list)
    assert len(body["top_words"]) <= 3
    for word_contrib in body["top_words"]:
        assert "word" in word_contrib
        assert "weight" in word_contrib


def test_explain_default_top_n(client):
    response = client.post("/explain", json={"text": "Great value for the price"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["top_words"]) <= 5
