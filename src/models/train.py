"""training.py: frozen extractor + logistic-regression head, conformal calibration, MLflow tracking,
registry and champion/challenger promotion.

Usage:
  python -m src.models.train                                   # train with params.yaml, promote if better
  python -m src.models.train --compare mobilenetv3_large_100 efficientnet_b0 resnet50
                                                               # same protocol for each extractor, no promotion
"""
import argparse
import hashlib
import json
import logging
import time

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import extract
from src.models import conformal

log = logging.getLogger("train")


def dataset_version() -> str:
    """DVC hash of data/raw when it is tracked, else the md5 of the metadata file."""
    dvc_file = config.DATA_DIR / "raw.dvc"
    if dvc_file.exists():
        for line in dvc_file.read_text().splitlines():
            if "md5:" in line:
                return line.split("md5:")[1].strip()
    return hashlib.md5(config.METADATA_CSV.read_bytes()).hexdigest()


def split(X, y, p):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=p["test_size"], stratify=y, random_state=p["seed"])
    rel = p["calib_size"] / (1 - p["test_size"])
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_tr, y_tr, test_size=rel, stratify=y_tr, random_state=p["seed"])
    return X_tr, X_cal, X_te, y_tr, y_cal, y_te


def evaluate(head, X, y, qhat, min_conf) -> dict:
    classes = list(head.classes_)
    probs = head.predict_proba(X)
    y_idx = np.array([classes.index(c) for c in y])
    sets = conformal.prediction_sets(probs, qhat)
    review = np.array([conformal.needs_review(r, qhat, min_conf) for r in probs])
    pred = np.array(classes)[probs.argmax(1)]
    auto = ~review
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "top3_accuracy": top_k_accuracy_score(y_idx, probs, k=min(3, len(classes)), labels=range(len(classes))),
        "conformal_coverage": conformal.coverage(sets, y_idx),
        "mean_set_size": float(sets.sum(1).mean()),
        "review_rate": float(review.mean()),
        # accuracy on what the service answers without a human: the number users actually experience
        "auto_accuracy": float((pred[auto] == y[auto]).mean()) if auto.any() else 0.0,
    }


def benchmark(extractor_name: str, image_paths: list[str], pretrained: bool) -> dict:
    """Service-cost metrics for the extractor: single-image CPU latency, size, parameter count."""
    from PIL import Image

    ex = extract.Extractor(extractor_name, pretrained=pretrained)
    ims = [Image.open(config.ROOT / p) for p in image_paths]
    ex.embed(ims[:1])  # warm-up
    t = []
    for im in ims:
        t0 = time.perf_counter()
        ex.embed([im])
        t.append((time.perf_counter() - t0) * 1000)
    return {"latency_ms_p50": float(np.median(t)), "latency_ms_p95": float(np.percentile(t, 95)),
            "params_millions": ex.n_params / 1e6, "weights_mb": ex.n_params * 4 / 2**20,
            "embedding_dim": ex.dim}


def champion_info(client: MlflowClient):
    try:
        mv = client.get_model_version_by_alias(config.REGISTERED_MODEL, config.CHAMPION_ALIAS)
    except Exception:
        return None
    info = mlflow.models.get_model_info(f"models:/{config.REGISTERED_MODEL}@{config.CHAMPION_ALIAS}")
    return mv, info.metadata or {}


def train(extractor_name: str | None = None, promote: bool = True, pretrained: bool = True,
          bench_images: int = 20) -> dict:
    prm = config.params()
    p, s = prm["train"], prm["serve"]
    extractor_name = extractor_name or prm["features"]["extractor"]
    extract.build(extractor_name, pretrained=pretrained)
    X, y, meta = extract.dataset(extractor_name)
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = split(X, y, p)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{extractor_name}-{time.strftime('%Y%m%d-%H%M')}") as run:
        mlflow.log_params({"extractor": extractor_name, "n_classes": len(set(y)), "n_train": len(y_tr),
                           "n_calib": len(y_cal), "n_test": len(y_te), "C": p["C"], "alpha": s["alpha"],
                           "min_confidence": s["min_confidence"], "seed": p["seed"]})
        mlflow.set_tags({"dataset_version": dataset_version(), "pretrained": pretrained})

        t0 = time.perf_counter()
        head = make_pipeline(StandardScaler(), LogisticRegression(
            C=p["C"], max_iter=p["max_iter"], class_weight="balanced"))  # balanced: long-tail ready
        head.fit(X_tr, y_tr)
        mlflow.log_metric("head_fit_seconds", time.perf_counter() - t0)

        classes = list(head.classes_)
        qhat = conformal.calibrate(head.predict_proba(X_cal), np.array([classes.index(c) for c in y_cal]),
                                   s["alpha"])
        metrics = {f"test_{k}": v for k, v in evaluate(head, X_te, y_te, qhat, s["min_confidence"]).items()}
        metrics["qhat"] = qhat
        if bench_images:
            metrics.update(benchmark(extractor_name, meta.local_path.head(bench_images).tolist(), pretrained))
        mlflow.log_metrics(metrics)
        mlflow.log_text(classification_report(y_te, head.predict(X_te), zero_division=0), "classification_report.txt")

        # Attribution travels with the model: every training image's author and license.
        mlflow.log_text(meta[["photo_id", "taxon_name", "license_code", "attribution", "url"]].to_csv(index=False),
                        "training_data_attribution.csv")

        model_meta = {"extractor": extractor_name, "classes": classes, "qhat": qhat, "alpha": s["alpha"],
                      "min_confidence": s["min_confidence"], "dataset_version": dataset_version()}
        mlflow.sklearn.log_model(head, "model", metadata=model_meta,
                                 registered_model_name=config.REGISTERED_MODEL if promote else None)
        log.info("run %s: %s", run.info.run_id, json.dumps({k: round(v, 4) for k, v in metrics.items()}))
        if promote:
            metrics["promoted"] = promote_if_better(run.info.run_id, head, extractor_name, classes,
                                                    X_te, y_te, metrics, s, p["min_macro_f1"])
    return metrics


def promote_if_better(run_id, head, extractor_name, classes, X_te, y_te, metrics, s, min_f1) -> bool:
    """Champion/challenger: on the same test set when comparable, else on logged metrics."""
    client = MlflowClient()
    version = next(v for v in client.search_model_versions(f"name='{config.REGISTERED_MODEL}'")
                   if v.run_id == run_id)
    new_f1 = metrics["test_macro_f1"]
    champ = champion_info(client)
    if champ is None:
        reason, better = "no champion yet", True
    else:
        mv, cmeta = champ
        if cmeta.get("extractor") == extractor_name and cmeta.get("classes") == classes:
            old = mlflow.sklearn.load_model(f"models:/{config.REGISTERED_MODEL}@{config.CHAMPION_ALIAS}")
            old_f1 = evaluate(old, X_te, y_te, cmeta["qhat"], s["min_confidence"])["macro_f1"]
            reason = f"same test set: new {new_f1:.4f} vs champion v{mv.version} {old_f1:.4f}"
        elif set(cmeta.get("classes", [])) != set(classes):
            old_f1 = -1.0  # the perimeter changed (new taxon): the old model cannot answer for it
            reason = "perimeter changed"
        else:
            old_f1 = client.get_run(mv.run_id).data.metrics.get("test_macro_f1", 0.0)
            reason = f"logged metrics: new {new_f1:.4f} vs champion v{mv.version} {old_f1:.4f}"
        better = new_f1 >= old_f1
    promoted = better and new_f1 >= min_f1
    client.set_model_version_tag(config.REGISTERED_MODEL, version.version, "decision",
                                 f"{'promoted' if promoted else 'rejected'} ({reason})")
    if promoted:
        client.set_registered_model_alias(config.REGISTERED_MODEL, config.CHAMPION_ALIAS, version.version)
    log.info("version %s %s (%s)", version.version, "PROMOTED" if promoted else "not promoted", reason)
    return promoted


def compare(extractors: list[str]) -> pd.DataFrame:
    rows = [{"extractor": e, **train(e, promote=False)} for e in extractors]
    df = pd.DataFrame(rows)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(config.REPORTS_DIR / "extractor_comparison.csv", index=False)
    print(df[["extractor", "test_macro_f1", "test_review_rate", "latency_ms_p50", "weights_mb"]].to_string())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    ap.add_argument("--no-promote", action="store_true")
    a = ap.parse_args()
    compare(a.compare) if a.compare else train(a.extractor, promote=not a.no_promote)
