"""Weekly training pipeline: ingest -> embed -> version data (DVC) -> train/compare/promote -> reload API.

Also triggered by: POST /train on the API, the Grafana alert webhook, and the monitoring DAG.

Project code runs in its own virtualenv (/opt/venv) through BashOperator: Airflow 2.x pins
SQLAlchemy 1.4 while the project uses 2.0, so the two never share an interpreter.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

RUN = "cd /opt/project && /opt/venv/bin/python -m "

with DAG(
    dag_id="species_training",
    start_date=datetime(2026, 1, 1),
    schedule="0 3 * * 1",  # Mondays 03:00
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["species-id", "training"],
) as dag:
    ingest = BashOperator(task_id="ingest", bash_command=RUN + "src.data.ingest")
    features = BashOperator(task_id="extract_features", bash_command=RUN + "src.features.extract")
    version_data = BashOperator(
        task_id="version_data",
        bash_command=(
            "cd /opt/project && export PATH=/opt/venv/bin:$PATH && "
            "([ -d .dvc ] || dvc init --no-scm -q) && dvc config cache.type hardlink,symlink && "
            "dvc add data/raw -q && "
            "(if [ -n \"$(dvc remote list)\" ]; then dvc push -q; else echo 'no DVC remote, skipping push'; fi)"
        ),
    )
    train = BashOperator(task_id="train", bash_command=RUN + "src.models.train")
    # Idempotent: the API reloads whatever version holds the "champion" alias.
    reload_api = BashOperator(
        task_id="reload_api",
        bash_command='curl -fsS -X POST -u "$API_ADMIN_USER:$API_ADMIN_PASSWORD" "$API_URL/reload"',
    )

    ingest >> features >> version_data >> train >> reload_api
