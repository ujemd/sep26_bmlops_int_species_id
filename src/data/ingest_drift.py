"""Daily sample of fresh iNaturalist photos sent to the production API, as monitoring traffic.

Separate from training ingestion:
- never writes images to data/raw and never inserts into observations/photos;
- skips photos already used for training, and images already predicted;
- calls POST /predict, which stores the prediction and its embedding (input of the drift check);
- then POST /feedback with the community's research-grade species (source="inat"), so the
  monitoring DAG can measure accuracy in production;
- keeps license, attribution and the link to the prediction in data/drift/metadata.csv.

Usage: python -m src.data.ingest_drift [--per-taxon N] [--seed S]
"""
import argparse
import csv
import hashlib
import logging
import random
from datetime import datetime, timezone

import requests
from sqlalchemy import select

from src import config
from src.data.ingest import HTTP, api_get, is_open_photo, resolve_taxon, sized_url
from src.db import Photo, Prediction, session

log = logging.getLogger("ingest_drift")

MANIFEST = config.DATA_DIR / "drift" / "metadata.csv"
FIELDS = ["collected_at_utc", "requested_taxon", "observed_taxon", "observation_id", "photo_id",
          "license_code", "attribution", "photo_url", "image_sha256", "prediction_id", "predicted_taxon",
          "confidence", "needs_review", "model_version", "feedback_recorded"]


def known_photos_and_images() -> tuple[set[int], set[str]]:
    """Photo ids used for training, and hashes of images already predicted."""
    with session() as s:
        return set(s.scalars(select(Photo.id)).all()), set(s.scalars(select(Prediction.image_sha256)).all())


def recent_observations(taxon_name: str, p: dict) -> tuple[str, list[dict]]:
    """The 200 most recently uploaded research-grade, openly licensed observations of a taxon."""
    taxon_id, canonical = resolve_taxon(taxon_name)
    results = api_get("observations", taxon_id=taxon_id, quality_grade="research", photos="true",
                      photo_license=",".join(p["allowed_licenses"]), order_by="created_at", order="desc",
                      per_page=200)["results"]
    return canonical, results


def append_manifest(row: dict):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    new_file = not MANIFEST.exists() or MANIFEST.stat().st_size == 0
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def predict_and_label(image: bytes, photo_id: int, true_taxon: str) -> tuple[dict, bool] | None:
    """POST /predict, then POST /feedback with the community identification.
    Returns (prediction, feedback_recorded), or None if the prediction failed."""
    api = config.API_URL.rstrip("/")
    try:
        r = HTTP.post(f"{api}/predict", files={"file": (f"inat-{photo_id}.jpg", image, "image/jpeg")}, timeout=90)
        if r.status_code != 200:
            log.warning("prediction failed photo_id=%s: HTTP %s %s", photo_id, r.status_code, r.text[:200])
            return None
        pred = r.json()
        fb = HTTP.post(f"{api}/feedback", timeout=30,
                       json={"prediction_id": pred["prediction_id"], "true_taxon": true_taxon, "source": "inat"})
        return pred, fb.status_code == 200
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning("API call failed photo_id=%s: %s", photo_id, e)
        return None


def run(per_taxon: int = 1, seed: int | None = None) -> int:
    if per_taxon < 1:
        raise ValueError("--per-taxon must be >= 1")
    p = config.params()["data"]
    allowed, host = set(p["allowed_licenses"]), p["open_data_host"]
    training_ids, known_hashes = known_photos_and_images()
    rng = random.Random(seed)
    total = 0
    for taxon_name in p["taxa"]:
        canonical, observations = recent_observations(taxon_name, p)
        rng.shuffle(observations)  # a random sample among the latest uploads
        added = 0
        for obs in observations:
            if added >= per_taxon:
                break
            photos = [ph for ph in obs.get("photos", []) if is_open_photo(ph, allowed, host)]
            if not photos or int(photos[0]["id"]) in training_ids:
                continue  # never use training photos as production traffic
            photo, photo_id = photos[0], int(photos[0]["id"])
            url = sized_url(photo["url"], p["photo_size"])
            try:
                r = HTTP.get(url, timeout=30)
                r.raise_for_status()
            except requests.RequestException as e:
                log.warning("download failed photo_id=%s: %s", photo_id, e)
                continue
            image = r.content
            digest = hashlib.sha256(image).hexdigest()
            if not image or digest in known_hashes:
                continue  # already predicted
            result = predict_and_label(image, photo_id, canonical)
            if result is None:
                continue
            pred, labelled = result
            known_hashes.add(digest)
            append_manifest({
                "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "requested_taxon": canonical,
                "observed_taxon": (obs.get("taxon") or {}).get("name", ""),
                "observation_id": obs.get("id", ""),
                "photo_id": photo_id,
                "license_code": (photo.get("license_code") or "").lower(),
                "attribution": photo.get("attribution", ""),
                "photo_url": url,
                "image_sha256": digest,
                "prediction_id": pred["prediction_id"],
                "predicted_taxon": pred.get("taxon", ""),
                "confidence": pred.get("confidence", ""),
                "needs_review": pred.get("needs_human_review", ""),
                "model_version": pred.get("model_version", ""),
                "feedback_recorded": labelled,
            })
            added += 1
        log.info("%-28s +%d monitoring predictions (requested %d)", canonical, added, per_taxon)
        total += added
    log.info("monitoring sample done: %d predictions", total)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-taxon", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    run(per_taxon=a.per_taxon, seed=a.seed)
