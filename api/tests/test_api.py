"""API tests. Run from the repository root:  pytest api/tests -v"""
import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "model" / "test_files" / "valid" / "grid_2024-10-22_18h_demo.csv"


def load_rows() -> list[dict]:
    with open(SAMPLE, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs the startup that loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_version(client):
    assert client.get("/version").json()["model_version"] == "1.0.0"


def test_predict_success(client):
    r = client.post("/predict", json={"rows": load_rows()})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 12
    assert body["best_window_avg_co2"] <= body["run_now_avg_co2"]
    assert body["latency_ms"] < 1000  # requirement: inference under 1 s on CPU


def test_predict_invalid_input_returns_422(client):
    rows = load_rows()
    rows[3]["consumption_mw"] = -500  # negative consumption is impossible
    assert client.post("/predict", json={"rows": rows}).status_code == 422


def test_predict_wrong_row_count_returns_422(client):
    assert client.post("/predict", json={"rows": load_rows()[:20]}).status_code == 422


def test_predict_missing_body_returns_422(client):
    assert client.post("/predict").status_code == 422


def test_metrics_counts_calls(client):
    body = client.get("/metrics").json()
    assert body["calls"] >= 1
    assert body["avg_latency_ms"] > 0
