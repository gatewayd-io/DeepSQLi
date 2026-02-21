import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

os.environ.setdefault("VOCAB_PATH", os.path.join(
    os.path.dirname(__file__), "..", "training", "sql_tokenizer_vocab.json"))
os.environ.setdefault("MODEL_PATH", os.path.join(
    os.path.dirname(__file__), "..", "sqli_model", "3"))

spec = importlib.util.spec_from_file_location(
    "api_module", os.path.join(os.path.dirname(__file__), "api.py"))
api_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_module)
app = api_module.app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_predict_missing_body(client):
    resp = client.post("/predict", content_type="application/json")
    assert resp.status_code == 400


def test_predict_missing_query_key(client):
    resp = client.post("/predict", json={"foo": "bar"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


def test_predict_sqli(client):
    resp = client.post("/predict", json={"query": "SELECT * FROM users WHERE id=1 OR 1=1"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


def test_predict_legitimate(client):
    resp = client.post("/predict", json={"query": "SELECT name FROM products"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


def test_predict_empty_query(client):
    resp = client.post("/predict", json={"query": ""})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "confidence" in data


def test_predict_error_not_leaked(client):
    """Ensure internal error details are not exposed to the client."""
    resp = client.post("/predict", json={"query": ""})
    if resp.status_code == 500:
        data = resp.get_json()
        assert data["error"] == "Internal server error"
