.PHONY: help venv test lint ingest features train compare predict drift up up-monitoring down logs train-dag clean-docker

PY ?= .venv/bin/python
IMG ?=

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

venv:            ## local Python 3.11 env with CPU-only torch
	python3.11 -m venv .venv
	.venv/bin/pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -r requirements/dev.txt

test:            ## unit tests (no network, no torch weights needed)
	$(PY) -m pytest -q

lint:            ## flake8
	$(PY) -m flake8 src tests airflow streamlit_app --max-line-length=127

ingest:          ## download openly licensed research-grade photos from iNaturalist
	$(PY) -m src.data.ingest

features:        ## embed new photos with the frozen extractor
	$(PY) -m src.features.extract

train:           ## train, log to MLflow, promote if better than the champion
	$(PY) -m src.models.train

compare:         ## level 2: same protocol, three extractors, cost vs accuracy table
	$(PY) -m src.models.train --compare mobilenetv3_large_100 efficientnet_b0 resnet50

predict:         ## make predict IMG=path/to/photo.jpg
	$(PY) -m src.models.predict $(IMG)

drift:           ## run the drift / production-performance check once
	$(PY) -m src.monitoring.drift

up:              ## build and start the core stack (postgres, mlflow, api, streamlit, airflow)
	docker compose build airflow
	docker compose build api
	docker compose up -d --build

up-monitoring:   ## add Prometheus + Grafana
	docker compose --profile monitoring up -d

down:            ## stop the stack (volumes kept)
	docker compose --profile monitoring down

logs:
	docker compose logs -f --tail=100

train-dag:       ## trigger the Airflow training DAG through the API
	curl -s -X POST -u admin:$${API_ADMIN_PASSWORD:-change-me} localhost:8000/train

clean-docker:    ## reclaim disk: dangling images and build cache
	docker image prune -f && docker builder prune -f
