# One-week roadmap

The code for every phase is already scaffolded and unit-tested. Each day is about **running it
on real data, fixing what breaks, and capturing evidence** (screenshots, MLflow runs, Grafana
panels) for the README and the defense.

| Day | Goal | Done when | Course phase |
|---|---|---|---|
| 1 | Local baseline on real data | `make venv ingest features train` runs; MLflow UI shows a run with `test_auto_accuracy`; `make predict` works on a photo of your own | 1 |
| 2 | Stack in Docker | `make up`; Streamlit → API → PostgreSQL works; the `predictions` table fills | 1–2 |
| 3 | Automated training | `species_training` DAG green end to end; DVC pushes to DagsHub; a second run shows a promote/reject decision tag | 2 |
| 4 | Monitoring | Grafana dashboard live; drift simulated (upload ~60 out-of-perimeter photos, run `species_monitoring`) and it triggers training; Grafana alert fires the webhook | 3 |
| 5 | Level 2 and CI/CD | `make compare` table (accuracy vs latency vs size) in the README; GitHub Actions green; images on Docker Hub | 2 |
| 6 | Polish | README screenshots and comparison table; optional k3s deployment of the API | 3 |
| 7 | Buffer and rehearsal | 15 min presentation + 5 min demo, every member speaks | Defense |

## If time is short, cut in this order
1. k3s (keep the manifest, say it is untested).
2. Grafana alert webhook (the drift DAG already closes the loop).
3. Extractor comparison with three models: two are enough.

## Memory budget (3.8 GB RAM VM)

| Service | Approx. RAM |
|---|---|
| Airflow (scheduler + 1 webserver worker) | ~1.0 GB |
| Training task (extractor + images in batches) | ~1.0 GB peak |
| API (torch + model) | ~0.5 GB |
| MLflow, PostgreSQL, Streamlit | ~0.5 GB |
| Prometheus + Grafana (`make up-monitoring`) | ~0.3 GB |

Add a 4 GB swap file before the first `make up`; builds and training peaks will spill into it
instead of getting killed. If it is still too slow, run `make up` without monitoring while
training, and start Grafana only for monitoring work and the demo.

## Disk budget (18 GB available)

| Item | Approx. size |
|---|---|
| Docker images (API ~1.3 GB, Airflow ~3 GB, MLflow, Streamlit, Postgres, Grafana, Prometheus) | ~6.5 GB |
| Photos: 10 taxa × 150–400 × ~60 KB | 0.1–0.25 GB |
| DVC cache | ~0 (hardlinks, no copies) |
| Build cache | grows: run `make clean-docker` after builds |

## Design notes for the remaining level 3–4 items (for the Q&A)

**Label revisions.** On iNaturalist, a research-grade identification can change later. Plan:
the ingestion job also re-reads stored observations with `updated_since=<last run>` and
writes the change to a `label_revisions` table. A training photo whose label changed is relabelled
before the next training run. Past predictions are never rewritten: the new truth is recorded as
`feedback` with `source='inat'`, so production accuracy is recomputed against it, and the history
stays auditable.

**Active learning.** The review queue is ordered by the size of the conformal prediction set
(largest first), then by lowest confidence. These are the photos where a human label is worth
the most.

**Rare species.** BioCLIP (via `open_clip`) can classify zero-shot from a text prompt with the
species name. This is a possible fallback for a taxon with fewer than ~20 images.

## Pending

- [ ] **Calibrate `min_confidence`** on the calibration set: lowest threshold that keeps
      auto-accuracy ≥ 95% (currently fixed at 0.5 by hand; only matters when `qhat > 0.5`).
- [ ] **Security checklist** (before the defense):
      strong passwords in `.env`; remove the `admin` / `change-me` fallbacks in `docker-compose.yml`
      and `src/config.py`; Kubernetes Secret for k3s; one Docker Hub token per project.
- [ ] Extractor comparison (`--compare`), results table in the README.
- [ ] DagsHub remote for DVC and MLflow.

## Notes for the defense

- **Images vs embeddings:** training only needs embeddings, but images are kept (DVC) to change
  the extractor, reproduce embeddings, benchmark real latency and produce visual explanations.
  At scale, the DVC remote is the source of truth and machines pull only what they need.
- **Licenses:** CC0, BY, BY-SA, BY-NC, BY-NC-SA kept; ND excluded (no derivatives). Attribution
  travels with the data in `metadata.csv`. NC is fine for a course project.
- **Split:** hash-based so a photo never changes part as data grows; not exactly stratified
  (27 ± 5 test photos per species at 180 photos), the price of a fair champion comparison.
- **Taxa:** common, well-represented, several kingdoms; North American bias of iNaturalist users.
- **Extractor:** MobileNetV3 as the cheap baseline; bigger models must justify their cost.