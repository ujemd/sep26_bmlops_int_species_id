import csv

import numpy as np
import requests

from src.data import ingest_drift
from src.db import Prediction, session
from src.monitoring.drift import recent_predictions

HOST = "https://inaturalist-open-data.s3.amazonaws.com"


def test_drift_counts_each_image_once():
    emb = np.zeros(4, dtype=np.float16).tobytes()
    with session() as s:
        rows = [Prediction(model_version="1", predicted_taxon="x", confidence=0.9, prediction_set="[]",
                           needs_review=False, latency_ms=1.0, image_sha256=sha, embedding=emb)
                for sha in ("dup-a", "dup-a", "dup-b")]
        s.add_all(rows)
        s.commit()
        last_a = rows[1].id
    df = recent_predictions(days=7)
    df = df[df.image_sha256.isin(["dup-a", "dup-b"])]
    assert sorted(df.image_sha256) == ["dup-a", "dup-b"]
    assert df[df.image_sha256 == "dup-a"].id.item() == last_a  # the latest prediction is kept


class FakeResponse:
    def __init__(self, status=200, content=b"", payload=None):
        self.status_code, self.content, self._payload, self.text = status, content, payload, ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self):
        self.posts = []

    def get(self, url, timeout):
        return FakeResponse(content=url.encode())  # a distinct "image" per URL

    def post(self, url, timeout, files=None, json=None):
        self.posts.append((url, json))
        if url.endswith("/predict"):
            return FakeResponse(payload={"prediction_id": 7, "taxon": "Danaus plexippus", "confidence": 0.9,
                                         "needs_human_review": False, "model_version": "1"})
        return FakeResponse()


def photo(pid, license_code="cc-by", host=HOST):
    return {"id": pid, "license_code": license_code, "attribution": f"(c) author {pid}",
            "url": f"{host}/photos/{pid}/square.jpg"}


def test_collector_predicts_new_open_photos_and_records_the_true_species(monkeypatch, tmp_path):
    observations = [
        {"id": 1, "taxon": {"name": "Danaus plexippus plexippus"}, "photos": [photo(10)]},
        {"id": 2, "taxon": {"name": "Danaus plexippus"},
         "photos": [photo(20, None, "https://static.inaturalist.org")]},  # not openly licensed
        {"id": 3, "taxon": {"name": "Danaus plexippus"}, "photos": [photo(30)]},  # used for training
    ]
    http = FakeHTTP()
    monkeypatch.setattr(ingest_drift, "MANIFEST", tmp_path / "metadata.csv")
    monkeypatch.setattr(ingest_drift, "HTTP", http)
    monkeypatch.setattr(ingest_drift, "resolve_taxon", lambda name: (48662, "Danaus plexippus"))
    monkeypatch.setattr(ingest_drift, "api_get", lambda *a, **k: {"results": list(observations)})
    monkeypatch.setattr(ingest_drift, "known_photos_and_images", lambda: ({30}, set()))
    monkeypatch.setattr(ingest_drift.config, "params", lambda: {"data": {
        "taxa": ["Danaus plexippus"], "allowed_licenses": ["cc-by"],
        "open_data_host": "inaturalist-open-data.s3.amazonaws.com", "photo_size": "medium"}})

    assert ingest_drift.run(per_taxon=2, seed=0) == 1  # only photo 10 is eligible
    feedback = [body for url, body in http.posts if url.endswith("/feedback")]
    assert feedback == [{"prediction_id": 7, "true_taxon": "Danaus plexippus", "source": "inat"}]
    rows = list(csv.DictReader(open(tmp_path / "metadata.csv")))
    assert len(rows) == 1 and rows[0]["photo_id"] == "10" and rows[0]["attribution"] == "(c) author 10"
    assert rows[0]["photo_url"].endswith("/10/medium.jpg") and rows[0]["feedback_recorded"] == "True"
