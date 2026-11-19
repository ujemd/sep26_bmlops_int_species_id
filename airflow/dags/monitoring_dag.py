"""Daily monitoring pipeline: collect production traffic -> detect drift -> decide whether to retrain.

1. collect_drift_data: a few fresh research-grade iNaturalist photos per species are sent to the
   API (stored as predictions, with their true species as feedback). They stand in for real users.
2. check_drift_and_performance: Evidently drift on embeddings + accuracy on feedback. Its last
   stdout line, a JSON decision, is pushed to XCom.
3. retrain_needed / trigger_training: start the training DAG only if the decision says so.
"""
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

RUN = "cd /opt/project && /opt/venv/bin/python -m "


def _should_retrain(ti) -> bool:
    decision = json.loads(ti.xcom_pull(task_ids="check_drift_and_performance"))
    return decision["retrain"]  # False short-circuits: no retraining


with DAG(
    dag_id="species_monitoring",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["species-id", "monitoring"],
) as dag:
    collect = BashOperator(task_id="collect_drift_data", bash_command=RUN + "src.data.ingest_drift --per-taxon 2",
                           execution_timeout=timedelta(minutes=30))
    check = BashOperator(task_id="check_drift_and_performance", bash_command=RUN + "src.monitoring.drift")
    gate = ShortCircuitOperator(task_id="retrain_needed", python_callable=_should_retrain)
    retrain = TriggerDagRunOperator(task_id="trigger_training", trigger_dag_id="species_training",
                                    conf={"reason": "drift or accuracy drop"})

    collect >> check >> gate >> retrain
