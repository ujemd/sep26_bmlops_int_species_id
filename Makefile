.RECIPEPREFIX = >
.PHONY: help venv lock test lint ingest features train compare predict drift up up-monitoring down logs train-dag clean-docker

PY ?= uv run python
IMG ?=

help:            ## list targets
> @grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

venv:            ## create .venv from uv.lock (downloads Python 3.11 if needed)
> uv sync --locked

lock:            ## refresh uv.lock after editing pyproject.toml
> uv lock

test:            ## unit tests (no network, no model weights needed)
> $(PY) -m pytest -q

lint:            ## flake8
> $(PY) -m flake8 src tests airflow streamlit_app --max-line-length=127

ingest:          ## download openly licensed research-grade photos from iNaturalist
> $(PY) -m src.data.ingest

features:        ## embed new photos with the frozen extractor
> $(PY) -m src.features.extract

train:           ## train, log to MLflow, promote if better than the champion
> $(PY) -m src.models.train

compare:         ## level 2: same protocol, three extractors, cost vs accuracy table
> $(PY) -m src.models.train --compare mobilenetv3_large_100 efficientnet_b0 resnet50

predict:         ## make predict IMG=path/to/photo.jpg
> $(PY) -m src.models.predict $(IMG)

drift:           ## run the drift / production-performance check once
> $(PY) -m src.monitoring.drift

up:              ## build and start the core stack (postgres, mlflow, api, streamlit, airflow)
> docker compose build airflow
> docker compose build api
> docker compose up -d --build

up-monitoring:   ## add Prometheus + Grafana
> docker compose --profile monitoring up -d

down:            ## stop the stack (volumes kept)
> docker compose --profile monitoring down

logs:
> docker compose logs -f --tail=100

train-dag:       ## trigger the Airflow training DAG through the API (admin password read from .env)
> @pw=$$(grep -E '^API_ADMIN_PASSWORD=' .env 2>/dev/null | cut -d= -f2- | sed 's/[[:space:]]*#.*//'); curl -sS -w '\nHTTP %{http_code}\n' -X POST -u "admin:$${pw:-change-me}" localhost:8000/train

clean-docker:    ## reclaim disk: dangling images and build cache
> docker image prune -f && docker builder prune -f
