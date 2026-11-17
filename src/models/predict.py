"""predict.py: load the champion model and identify a photo.

Usage: python -m src.models.predict path/to/photo.jpg
"""
import io
import json
import logging
import sys
import time

import numpy as np

from src import config
from src.models import conformal

log = logging.getLogger("predict")


class Predictor:
    def __init__(self, model_uri: str | None = None, pretrained: bool = True):
        import mlflow

        from src.features.extract import Extractor

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        self.uri = model_uri or f"models:/{config.REGISTERED_MODEL}@{config.CHAMPION_ALIAS}"
        info = mlflow.models.get_model_info(self.uri)
        self.meta = info.metadata
        self.head = mlflow.sklearn.load_model(self.uri)
        self.classes = list(self.head.classes_)
        self.version = self._version()
        self.extractor = Extractor(self.meta["extractor"], pretrained=pretrained)
        self.top_k = config.params()["serve"]["top_k"]
        log.info("loaded %s (version %s, %s)", self.uri, self.version, self.meta["extractor"])

    def _version(self) -> str:
        if self.uri.startswith("models:/") and "@" in self.uri:
            from mlflow.tracking import MlflowClient
            name, alias = self.uri[len("models:/"):].split("@")
            return str(MlflowClient().get_model_version_by_alias(name, alias).version)
        return self.uri

    def predict(self, image_bytes: bytes) -> dict:
        from PIL import Image

        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes))
        emb = self.extractor.embed([img])
        probs = self.head.predict_proba(emb)[0]
        qhat, min_conf = self.meta["qhat"], self.meta["min_confidence"]
        order = np.argsort(probs)[::-1]
        pset = [self.classes[i] for i in order if probs[i] >= 1 - qhat]
        return {
            "taxon": self.classes[order[0]],
            "confidence": float(probs[order[0]]),
            "top_k": [{"taxon": self.classes[i], "probability": float(probs[i])} for i in order[: self.top_k]],
            "prediction_set": pset,
            "needs_human_review": conformal.needs_review(probs, qhat, min_conf),
            "model_version": self.version,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "_embedding": emb[0],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = Predictor().predict(open(sys.argv[1], "rb").read())
    out.pop("_embedding")
    print(json.dumps(out, indent=2))
