# Roadmap, budgets and notes for the defense

| Day | Goal | Status |
|---|---|---|
| 1 | Local baseline on real data | ✅ |
| 2 | Stack in Docker | ✅ |
| 3 | Automated training: Airflow, DVC + DagsHub, champion/challenger promotion | ✅ |
| 4 | Monitoring: daily collector, Evidently drift, Grafana dashboard and alert | ✅ |
| 5 | Extractor comparison; CI/CD (GitHub Actions, Docker Hub) | ✅ |
| 6 | README screenshots; optional k3s deployment of the API | ⏳ |
| 7 | Rehearsal: 15 min presentation + 5 min demo, every member speaks | ⏳ |

## Memory budget (3.8 GB RAM VM)

| Service | Approx. RAM |
|---|---|
| Airflow (scheduler + 1 webserver worker) | ~1.0 GB |
| Training task (extractor + images in batches) | ~1.0 GB peak |
| API (torch + model) | ~0.5 GB |
| PostgreSQL, Streamlit | ~0.3 GB |
| Prometheus + Grafana (`make up-monitoring`) | ~0.3 GB |

Before the first `make up` on the course VM:
- stop k3s until it is needed: `sudo systemctl disable k3s && sudo /usr/local/bin/k3s-killall.sh`
  (back with `sudo systemctl enable --now k3s`);
- add a 4 GB swap file with `vm.swappiness=10`.

Heavy jobs (extractor comparison, image builds): stop Streamlit and Grafana first if RAM is short.

## Disk budget (29 GB disk)

| Item | Approx. size |
|---|---|
| Docker images | ~6.5 GB |
| Local `.venv` and uv cache | ~2 GB |
| Swap file | 4 GB |
| Photos and embeddings | ~0.3 GB |
| DVC cache | ~0 (hardlinks) |

Rebuilding images: `make down`, remove the old `local/species-*` images, build one image at a time
and run `docker builder prune -f` after each; old and new images do not fit side by side.

## Pending

- [ ] **Promotion tolerance:** promote when the new model is at most 0.01 F1 below the champion
      (differences of a few thousandths are noise; today a tie or noise blocks promotion).
- [ ] **Calibrate `min_confidence`** on the calibration set (fixed at 0.5 by hand; redundant
      while `qhat < 0.5`).
- [ ] **Use feedback for training:** add corrected photos to the training data after a quality
      check (requires storing the images of predictions).
- [ ] **Review queue:** store flagged images, show them to reviewers ordered by uncertainty.
- [ ] **Shadow test of the challenger** on live traffic before promotion.
- [ ] **k3s deployment** of the API (manifest in `k8s/api.yaml`, images on Docker Hub).
- [ ] **CI check** that committed `.dvc` pointers exist on the remote (`dvc status -c`).
- [ ] **Security checklist:** strong passwords in `.env` (PostgreSQL also needs `ALTER USER`);
      remove the `admin` / `change-me` fallbacks; Kubernetes Secret; one token per project.

## Design notes for the remaining items (Q&A)

**Label revisions.** A research-grade identification can change later. The ingestion job would
re-read stored observations with `updated_since=<last run>` and record changes in a
`label_revisions` table. Training photos are relabelled before the next run; past predictions are
never rewritten: the new truth is added as feedback (`source='inat'`), so production accuracy is
recomputed and the history stays auditable.

**Photos that disappear or change license.** The same job checks stored photos (HTTP `HEAD`,
license field) and marks them `unavailable` or `license_revoked`; the next training run and the
next DVC version exclude them. A revoked license means models trained on that photo must be
retrained without it.

**Active learning.** Order the review queue by prediction-set size, then by lowest confidence,
then by out-of-scope distance: these are the photos where a human label is worth the most.

**Rare species.** BioCLIP (via `open_clip`) classifies zero-shot from the species name: a fallback
for taxa with fewer than ~20 photos.

## Notes for the defense

- **Extractor:** MobileNetV3 kept; ResNet-50 is 6.7× slower for +0.003 F1 (noise). The extractor
  is not the bottleneck at ~0.93 F1.
- **Two reasons to ask a human:** conformal sets (the model hesitates between species) and the
  out-of-scope detector (the photo is unlike anything in training). Softmax alone answers
  confidently for a cat; conformal guarantees assume in-distribution photos.
- **Look-alikes:** the viceroy is predicted as a monarch; only adding it to the perimeter fixes it.
- **Split:** stratified and stable (blocks of 20 by observation order); a photo never changes part,
  so the champion comparison stays fair.
- **Data versions:** MLflow records which data each model used (hashes + `.dvc` files); the DVC
  remote keeps every version; Git holds a convenient snapshot of the latest one.
- **Images vs metadata only:**

  | | Metadata + embeddings only | Metadata + photos (current) |
  |---|---|---|
  | Storage | tiny | ~60 KB per photo |
  | Retrain the head | ✅ | ✅ |
  | Change the extractor | ❌ re-download everything | ✅ |
  | Exact reproducibility | ❌ photos can be deleted or relicensed | ✅ |

- **Monitoring without users:** the daily collector sends fresh, community-labelled photos through
  the API, so drift and accuracy are measured on real, recent data.
- **Licenses:** CC0, BY, BY-SA, BY-NC, BY-NC-SA kept; ND excluded. Attribution travels with the
  data and with every model.
- **Security:** course-grade on purpose; the checklist above is what a real deployment needs.
- **Manual promotion:** used twice, both recorded with their reason in the `decision` tag.
