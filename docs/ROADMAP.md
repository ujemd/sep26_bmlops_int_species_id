# One-week roadmap

Each day is about **running the pipeline on real data, fixing what breaks, and capturing
evidence** (screenshots, MLflow runs, Grafana panels) for the README and the defense.

| Day | Goal | Done when | Course phase | Status |
|---|---|---|---|---|
| 1 | Local baseline on real data | ingestion and training run; MLflow shows a run with `test_auto_accuracy` | 1 | ✅ |
| 2 | Stack in Docker | Streamlit → API → PostgreSQL works; the `predictions` table fills | 1–2 | ✅ |
| 3 | Automated training | `species_training` DAG green end to end; DVC pushes to DagsHub; a second run shows a promote/reject decision tag | 2 | ✅ except DagsHub |
| 4 | Monitoring | Grafana dashboard live; drift simulated with `scripts/simulate_traffic.py` and it triggers training; Grafana alert fires the webhook | 3 | ⏳ next |
| 5 | Level 2 and CI/CD | extractor comparison table (accuracy vs latency vs size) in the README; GitHub Actions green; images on Docker Hub | 2 | CI/CD ✅, comparison ⏳ |
| 6 | Polish | README screenshots and comparison table; optional k3s deployment of the API | 3 | ⏳ |
| 7 | Buffer and rehearsal | 15 min presentation + 5 min demo, every member speaks | Defense | ⏳ |

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

Before the first `make up` on the course VM:
- stop k3s until the scaling step: `sudo systemctl disable k3s && sudo /usr/local/bin/k3s-killall.sh`
  (back with `sudo systemctl enable --now k3s`);
- add a 4 GB swap file with `vm.swappiness=10`, so build and training peaks spill to disk
  instead of getting killed.

## Disk budget (29 GB disk)

| Item | Approx. size |
|---|---|
| Docker images (API ~1.3 GB, Airflow ~3 GB, MLflow, Streamlit, Postgres, Grafana, Prometheus) | ~6.5 GB |
| Local `.venv` and uv download cache | ~2 GB |
| Swap file | 4 GB |
| Photos: 10 taxa × 150–400 × ~60 KB | 0.1–0.25 GB |
| DVC cache | ~0 (hardlinks, no copies) |
| Build cache | grows: run `make clean-docker` after builds |

Rebuilding images: stop the stack and remove the old project images first (`make down`, then
`docker image rm` on the four `local/species-*` images), build one image at a time, and run
`docker builder prune -f` after each build. Old and new images do not fit side by side.

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
- [ ] Extractor comparison (`make compare`), results table in the README.
- [ ] DagsHub remote for DVC and MLflow.
- [ ] Taxa justification with iNaturalist counts (`scripts/taxa_stats.py`) in the README.

## Notes for the defense

- **Images vs embeddings:** training only needs embeddings, but images are kept (DVC) to change
  the extractor, reproduce embeddings, benchmark real latency and produce visual explanations.
  At scale, the DVC remote is the source of truth and machines pull only what they need.
- **Licenses:** CC0, BY, BY-SA, BY-NC, BY-NC-SA kept; ND excluded (no derivatives). Attribution
  travels with the data in `metadata.csv`. NC is fine for a course project.
- **Split:** stratified *and* stable. Photos are ranked by observation id within each species and
  every block of 20 gets fixed slots (14 / 3 / 3). Ingestion only appends newer observations, so a
  photo never changes part and the champion comparison stays fair; a new species with 20 photos
  still gets 3 test photos.
- **Taxa:** common, well-represented, several kingdoms; North American bias of iNaturalist users.
- **Extractor:** MobileNetV3 as the cheap baseline; bigger models must justify their cost.
- **Environment:** one `uv.lock` for local, CI and every image; each image installs only its group.
