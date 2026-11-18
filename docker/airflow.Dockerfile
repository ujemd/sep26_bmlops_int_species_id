FROM ghcr.io/astral-sh/uv:X.Y.Z AS uv

FROM apache/airflow:2.10.5-python3.11
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

# Project dependencies live in their own venv (/opt/venv), isolated from Airflow's pinned packages
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock .python-version /tmp/project/
RUN cd /tmp/project && uv sync --locked --only-group train --python /usr/local/bin/python3.11 \
    && chmod -R a+rX /opt/venv
USER airflow
