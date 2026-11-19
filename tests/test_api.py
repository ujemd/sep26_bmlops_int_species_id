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


def test_model_loads_later_when_mlflow_was_down_at_startup(monkeypatch):
    """After a reboot the API can start before MLflow is reachable: it must keep retrying."""
    attempts = []

    def flaky_loader():
        attempts.append(1)
        main.STATE["last_attempt"] = main.time.monotonic()
        main.STATE["predictor"] = FakePredictor() if len(attempts) >= 2 else None

    monkeypatch.setattr(main, "load_predictor", flaky_loader)
    main.STATE.update(predictor=None, last_attempt=0.0)
    with TestClient(main.app) as c:  # startup: first attempt fails
        assert c.get("/health").json()["model_loaded"] is False  # too soon to retry
        assert c.post("/predict", files={"file": ("x.jpg", b"x")}).status_code == 503
        main.STATE["last_attempt"] -= main.RETRY_SECONDS + 1  # time passes
        assert c.get("/health").json()["model_loaded"] is True
    assert len(attempts) == 2
    main.STATE.update(predictor=None, last_attempt=0.0)
