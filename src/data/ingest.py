"""Incremental ingestion of research-grade, openly licensed iNaturalist photos.

- Metadata comes from the iNaturalist API v1 (keyless, 60 req/min -> we sleep 1.1 s between calls).
- Images come only from the AWS open-data bucket. A photo whose license is not in
  `allowed_licenses`, or whose URL is not on the open-data host, is rejected: photos on
  static.inaturalist.org are not openly licensed.
- License code and attribution are stored for every photo (DB + data/raw/metadata.csv).
- Each run only fetches observations newer than the last one stored per taxon (cursor = id_above),
  so a scheduled run adds fresh data instead of re-downloading everything.

Usage: python -m src.data.ingest [--per-taxon N]
"""
import argparse
import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
from sqlalchemy import func, select

from src import config
from src.db import Observation, Photo, session

log = logging.getLogger("ingest")
API = "https://api.inaturalist.org/v1"
HTTP = requests.Session()
HTTP.headers["User-Agent"] = config.USER_AGENT
_last_call = 0.0


def api_get(path: str, **query) -> dict:
    """GET on the iNat API, respecting the 60 requests/minute limit."""
    global _last_call
    wait = 1.1 - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        _last_call = time.monotonic()
        r = HTTP.get(f"{API}/{path}", params=query, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def resolve_taxon(name: str) -> tuple[int, str]:
    results = api_get("taxa", q=name, rank="species", per_page=5)["results"]
    for t in results:
        if t["name"].lower() == name.lower():
            return t["id"], t["name"]
    raise ValueError(f"Taxon not found on iNaturalist: {name}")


def is_open_photo(photo: dict, allowed: set[str], host: str) -> bool:
    lic = (photo.get("license_code") or "").lower()
    return lic in allowed and urlparse(photo.get("url") or "").hostname == host


def sized_url(url: str, size: str) -> str:
    # API returns .../photos/<id>/square.jpg ; other sizes share the same path.
    return url.replace("/square.", f"/{size}.")


def parse_observation(obs: dict, allowed: set[str], host: str, size: str):
    """Return (Observation, [Photo]) keeping only openly licensed photos (first one per observation)."""
    photos = [p for p in obs.get("photos", []) if is_open_photo(p, allowed, host)]
    if not photos or not obs.get("taxon"):
        return None, []
    lat = lng = None
    if obs.get("location"):
        lat, lng = (float(v) for v in obs["location"].split(","))
    o = Observation(
        id=obs["id"], taxon_id=obs["taxon"]["id"], taxon_name=obs["taxon"]["name"],
        quality_grade=obs.get("quality_grade", ""), observed_on=obs.get("observed_on"),
        latitude=lat, longitude=lng, user_login=(obs.get("user") or {}).get("login"),
    )
    p = photos[0]  # one photo per observation avoids near-duplicates leaking across splits
    return o, [Photo(id=p["id"], observation_id=obs["id"], license_code=p["license_code"].lower(),
                     attribution=p.get("attribution", ""), url=sized_url(p["url"], size))]


def download(photo: Photo, taxon_name: str) -> Photo | None:
    dest = config.IMAGES_DIR / taxon_name.replace(" ", "_") / f"{photo.id}.jpg"
    if not dest.exists():
        try:
            r = HTTP.get(photo.url, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("download failed %s: %s", photo.url, e)
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
    photo.local_path = str(dest.relative_to(config.ROOT))
    return photo


def ingest_taxon(name: str, n_new: int, p: dict) -> int:
    taxon_id, canonical = resolve_taxon(name)
    allowed, host, size = set(p["allowed_licenses"]), p["open_data_host"], p["photo_size"]
    with session() as s:
        cursor = s.scalar(select(func.max(Observation.id)).where(Observation.taxon_id == taxon_id)) or 0
        have = s.scalar(select(func.count()).select_from(Observation).where(Observation.taxon_id == taxon_id))
    n_new = min(n_new, p["max_per_taxon"] - have)
    added = 0
    while added < n_new:
        page = api_get("observations", taxon_id=taxon_id, quality_grade="research", photos="true",
                       photo_license=",".join(allowed), order_by="id", order="asc",
                       id_above=cursor, per_page=200)["results"]
        if not page:
            break
        cursor = page[-1]["id"]
        batch = []
        for obs in page:
            o, photos = parse_observation(obs, allowed, host, size)
            if o:
                # label = requested species, even when the observation is identified to a subspecies
                o.taxon_id, o.taxon_name = taxon_id, canonical
                batch.append((o, photos[0]))
        batch = batch[: n_new - added]
        with ThreadPoolExecutor(8) as pool:
            ok = list(pool.map(lambda op: (op[0], download(op[1], canonical)), batch))
        with session() as s:
            for o, ph in ok:
                if ph is not None:
                    s.merge(o)
                    s.flush()
                    s.merge(ph)
                    added += 1
            s.commit()
    log.info("%-28s +%d images (had %d)", canonical, added, have)
    return added


def export_metadata_csv():
    """Flat file versioned by DVC alongside the images: one row per photo, with attribution."""
    cols = ["photo_id", "observation_id", "taxon_id", "taxon_name", "license_code", "attribution",
            "url", "local_path", "observed_on", "latitude", "longitude"]
    with session() as s:
        rows = s.execute(select(Photo, Observation).join(Observation).order_by(Photo.id)).all()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.METADATA_CSV.unlink(missing_ok=True)
    with open(config.METADATA_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for ph, o in rows:
            w.writerow([ph.id, o.id, o.taxon_id, o.taxon_name, ph.license_code, ph.attribution, ph.url,
                        ph.local_path, o.observed_on, o.latitude, o.longitude])
    return len(rows)


def run(per_taxon: int | None = None) -> int:
    p = config.params()["data"]
    with session() as s:
        first_run = s.scalar(select(func.count()).select_from(Photo)) == 0
    n = per_taxon or (p["first_run_per_taxon"] if first_run else p["increment_per_taxon"])
    total = sum(ingest_taxon(name, n, p) for name in p["taxa"])
    rows = export_metadata_csv()
    log.info("ingestion done: +%d new images, %d in total", total, rows)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-taxon", type=int, default=None)
    run(ap.parse_args().per_taxon)
