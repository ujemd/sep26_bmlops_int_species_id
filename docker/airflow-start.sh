#!/usr/bin/env bash
# Single-container Airflow (scheduler + webserver, LocalExecutor): enough for one VM.
set -e
airflow db migrate
airflow users create -u "$AIRFLOW_USER" -p "$AIRFLOW_PASSWORD" -r Admin \
  -f Admin -l User -e admin@example.org 2>/dev/null || true
airflow scheduler &
exec airflow webserver
