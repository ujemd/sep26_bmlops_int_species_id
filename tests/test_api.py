import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import main


class FakePredictor:
    version = "3"
    classes = ["Apis mellifera", "Danaus plexippus"]
    meta = {"extractor": "fake", "alpha": 0.1, "dataset_version": "abc"}

    def predict(self, data):
        return {"taxon": "Danaus plexippus", "confidence": 0.93,
                "top_k": [{"taxon": "Danaus plexippus", "probability": 0.93}],
                "prediction_set": ["Danaus plexippus"], "needs_human_review": False,
                "model_version": "3", "latency_ms": 12.0, "_embedding": np.zeros(8)}


@pytest.fixture
def client():
    main.STATE["predictor"] = FakePredictor()
    with TestClient(main.app) as c:
        yield c
    main.STATE["predictor"] = None


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_then_feedback(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"fake-bytes")})
    assert r.status_code == 200
    body = r.json()
    assert body["taxon"] == "Danaus plexippus" and not body["needs_human_review"]
    r = client.post("/feedback", json={"prediction_id": body["prediction_id"], "true_taxon": "Apis mellifera"})
    assert r.json() == {"status": "recorded"}
    assert "species_predictions_total" in client.get("/metrics").text


def test_feedback_unknown_prediction(client):
    assert client.post("/feedback", json={"prediction_id": 999999, "true_taxon": "x"}).status_code == 404


def test_admin_routes_require_auth(client):
    assert client.post("/train").status_code == 401
    assert client.post("/train", auth=("admin", "wrong")).status_code == 401


def test_grafana_resolved_alert_is_ignored(client):
    r = client.post("/webhooks/grafana", json={"status": "resolved"},
                    auth=(config.API_ADMIN_USER, config.API_ADMIN_PASSWORD))
    assert r.json() == {"status": "ignored"}
