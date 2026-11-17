FROM apache/airflow:2.10.5-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/*
# Project dependencies live in their own venv, isolated from Airflow's pinned packages
RUN python -m venv /opt/venv && chown -R airflow /opt/venv
USER airflow
RUN /opt/venv/bin/pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 \
        --index-url https://download.pytorch.org/whl/cpu
COPY requirements/ /tmp/requirements/
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements/train.txt
