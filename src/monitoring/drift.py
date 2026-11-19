"""Drift and production-performance check, run daily by Airflow.

Reference = training embeddings of the champion's extractor. Current = embeddings of photos
received by the API over the last `drift_window_days`. Both are projected on the same 10 PCA
components (fitted on the reference) and compared column by column with Evidently.
Production accuracy comes from predictions that later received a ground-truth label (feedback).

Usage: python -m src.monitoring.drift   -> prints a JSON decision {"retrain": bool, ...}
"""
import json
import logging
from datetime import datetime, timedelta, timezone

import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sqlalchemy import select

from src import config
from src.db import Feedback, Prediction, session
from src.features.extract import dataset

log = logging.getLogger("drift")


def recent_predictions(days: int) -> pd.DataFrame:
    """Predictions of the last `days` days, each distinct image counted once (its latest prediction)."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with session() as s:
        rows = s.execute(select(Prediction).where(Prediction.created_at >= since)).scalars().all()
    df = pd.DataFrame([{"id": r.id, "image_sha256": r.image_sha256, "confidence": r.confidence,
                        "needs_review": r.needs_review, "predicted": r.predicted_taxon,
                        "embedding": np.frombuffer(r.embedding, dtype=np.float16).astype(np.float32)}
                       for r in rows if r.embedding])
    if df.empty:
        return df
    return df.sort_values("id").drop_duplicates(subset="image_sha256", keep="last").reset_index(drop=True)


def feedback_accuracy(days: int) -> tuple[float | None, int]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with session() as s:
        rows = s.execute(select(Prediction.predicted_taxon, Feedback.true_taxon)
                         .join(Feedback, Feedback.prediction_id == Prediction.id)
                         .where(Feedback.created_at >= since)).all()
    if not rows:
        return None, 0
    return float(np.mean([p == t for p, t in rows])), len(rows)


def embedding_drift(ref: np.ndarray, cur: np.ndarray, n_components: int):
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    pca = PCA(n_components=n_components, random_state=0).fit(ref)
    cols = [f"pc{i}" for i in range(n_components)]
    ref_df = pd.DataFrame(pca.transform(ref), columns=cols)
    cur_df = pd.DataFrame(pca.transform(cur), columns=cols)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    result = report.as_dict()["metrics"][0]["result"]
    return result["share_of_drifted_columns"], report


def run() -> dict:
    m = config.params()["monitoring"]
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT + "-monitoring")
    cur = recent_predictions(m["drift_window_days"])
    acc, n_fb = feedback_accuracy(m["drift_window_days"])
    out = {"n_predictions": len(cur), "n_feedback": n_fb, "feedback_accuracy": acc,
           "drift_share": None, "drift": False, "retrain": False}

    with mlflow.start_run(run_name=f"monitoring-{datetime.now():%Y%m%d-%H%M}"):
        if len(cur) >= m["min_current_samples"]:
            info = mlflow.models.get_model_info(f"models:/{config.REGISTERED_MODEL}@{config.CHAMPION_ALIAS}")
            X_ref, _, _ = dataset(info.metadata["extractor"])
            share, report = embedding_drift(X_ref, np.stack(cur.embedding), m["pca_components"])
            out["drift_share"], out["drift"] = share, share >= m["drift_share_threshold"]
            path = config.REPORTS_DIR / "drift_report.html"
            config.REPORTS_DIR.mkdir(exist_ok=True)
            report.save_html(str(path))
            mlflow.log_artifact(str(path))
            mlflow.log_metrics({"drift_share": share, "review_rate": float(cur.needs_review.mean()),
                                "mean_confidence": float(cur.confidence.mean())})
        if acc is not None:
            mlflow.log_metric("feedback_accuracy", acc)
        out["retrain"] = out["drift"] or (acc is not None and n_fb >= 20 and acc < m["min_feedback_accuracy"])
        mlflow.log_dict(out, "decision.json")
    log.info(json.dumps(out))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run()))
