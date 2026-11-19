"""Inference API.

Public : GET /health, GET /model, POST /predict, POST /feedback, GET /metrics (Prometheus)
Admin  : POST /train, POST /reload, POST /webhooks/grafana   (HTTP basic auth)
"""
import hashlib
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager

import numpy as np
import requests
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src import config
from src.db import Feedback, Prediction, session

log = logging.getLogger("api")
STATE: dict = {"predictor": None, "last_attempt": 0.0}
RETRY_SECONDS = 30  # while no model is loaded, try again at most this often

PREDICTIONS = Counter("species_predictions_total", "Predictions served", ["needs_review"])
CONFIDENCE = Histogram("species_prediction_confidence", "Top-1 probability",
                       buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
INFER_LATENCY = Histogram("species_inference_seconds", "Model inference time",
                          buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5])
FEEDBACK = Counter("species_feedback_total", "Ground-truth labels received", ["correct"])
MODEL_VERSION = Gauge("species_model_version", "Champion version currently served")


def load_predictor():
    from src.models.predict import Predictor
    STATE["last_attempt"] = time.monotonic()
    try:
        STATE["predictor"] = Predictor()
        MODEL_VERSION.set(float(STATE["predictor"].version))
    except Exception as e:  # the API must start even before the first model exists
        log.warning("no model loaded: %s", e)
        STATE["predictor"] = None


def ensure_model():
    """Load the champion if none is loaded yet, e.g. MLflow was not reachable when the API started
    (after a reboot) or no model had been trained yet. Retried at most every RETRY_SECONDS."""
    if STATE["predictor"] is None and time.monotonic() - STATE["last_attempt"] > RETRY_SECONDS:
        load_predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if STATE["predictor"] is None:
        load_predictor()
    yield


app = FastAPI(
    title="Species identification API", version="0.1.0", lifespan=lifespan,
    description="Identify a species from a photo, with a calibrated confidence and a referral to human "
                "review when the model is unsure. Admin routes require HTTP basic auth.",
)
Instrumentator().instrument(app).expose(app, include_in_schema=False)
security = HTTPBasic()


def admin(creds: HTTPBasicCredentials = Depends(security)):
    ok = secrets.compare_digest(creds.username, config.API_ADMIN_USER) and \
        secrets.compare_digest(creds.password, config.API_ADMIN_PASSWORD)
    if not ok:
        raise HTTPException(401, "Invalid credentials", headers={"WWW-Authenticate": "Basic"})


def predictor():
    ensure_model()
    if STATE["predictor"] is None:
        raise HTTPException(503, "No model in production yet: run the training pipeline first")
    return STATE["predictor"]


@app.get("/health", tags=["service"])
def health():
    """Liveness check, and whether a model is loaded (retries loading it if not)."""
    ensure_model()
    return {"status": "ok", "model_loaded": STATE["predictor"] is not None}


@app.get("/model", tags=["service"])
def model_info(p=Depends(predictor)):
    """Version, extractor, species and dataset version of the model in production."""
    return {"version": p.version, "extractor": p.meta["extractor"], "classes": p.classes,
            "alpha": p.meta["alpha"], "dataset_version": p.meta.get("dataset_version")}


@app.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(...), p=Depends(predictor)):
    """Identify the species in a photo: top-k, conformal prediction set, and whether a human should
    review it. Each call is stored for monitoring."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        out = p.predict(data)
    except Exception as e:
        raise HTTPException(400, f"Unreadable image: {e}")
    emb = out.pop("_embedding")
    PREDICTIONS.labels(needs_review=str(out["needs_human_review"]).lower()).inc()
    CONFIDENCE.observe(out["confidence"])
    INFER_LATENCY.observe(out["latency_ms"] / 1000)
    with session() as s:
        row = Prediction(model_version=out["model_version"], predicted_taxon=out["taxon"],
                         confidence=out["confidence"], prediction_set=json.dumps(out["prediction_set"]),
                         needs_review=out["needs_human_review"], latency_ms=out["latency_ms"],
                         image_sha256=hashlib.sha256(data).hexdigest(),
                         embedding=np.asarray(emb, dtype=np.float16).tobytes())
        s.add(row)
        s.commit()
        out["prediction_id"] = row.id
    return out


class FeedbackIn(BaseModel):
    prediction_id: int
    true_taxon: str
    source: str = "user"


@app.post("/feedback", tags=["prediction"])
def feedback(fb: FeedbackIn):
    """Record the true species for a past prediction (user correction or expert review)."""
    with session() as s:
        pred = s.get(Prediction, fb.prediction_id)
        if pred is None:
            raise HTTPException(404, "Unknown prediction")
        s.add(Feedback(prediction_id=fb.prediction_id, true_taxon=fb.true_taxon, source=fb.source))
        s.commit()
        FEEDBACK.labels(correct=str(pred.predicted_taxon == fb.true_taxon).lower()).inc()
    return {"status": "recorded"}


def trigger_training(reason: str) -> dict:
    r = requests.post(f"{config.AIRFLOW_URL}/api/v1/dags/species_training/dagRuns",
                      json={"conf": {"reason": reason}},
                      auth=(config.AIRFLOW_USER, config.AIRFLOW_PASSWORD), timeout=10)
    if r.status_code >= 300:
        raise HTTPException(502, f"Airflow refused the run: {r.text[:200]}")
    return {"status": "training triggered", "dag_run_id": r.json().get("dag_run_id")}


@app.post("/train", tags=["admin"], dependencies=[Depends(admin)])
def train():
    """Trigger the Airflow training pipeline."""
    return trigger_training("manual /train call")


@app.post("/webhooks/grafana", tags=["admin"], dependencies=[Depends(admin)])
def grafana_webhook(payload: dict):
    """Grafana alert webhook: a firing alert triggers retraining."""
    if payload.get("status") != "firing":
        return {"status": "ignored"}
    return trigger_training(f"grafana alert: {payload.get('title', '')}")


@app.post("/reload", tags=["admin"], dependencies=[Depends(admin)])
def reload():
    """Reload the model that holds the champion alias."""
    load_predictor()
    return {"status": "ok", "model_loaded": STATE["predictor"] is not None}
