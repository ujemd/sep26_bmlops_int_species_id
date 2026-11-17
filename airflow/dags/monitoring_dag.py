"""Daily monitoring: Evidently embedding drift + production accuracy from feedback.
Triggers the training DAG when drift is detected or accuracy falls below the threshold."""
import json
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


def _should_retrain(ti) -> bool:
    decision = json.loads(ti.xcom_pull(task_ids="check_drift_and_performance"))
    return decision["retrain"]  # False short-circuits: no retraining


with DAG(
    dag_id="species_monitoring",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    tags=["species-id", "monitoring"],
) as dag:
    # The last stdout line (the JSON decision) is pushed to XCom.
    check = BashOperator(task_id="check_drift_and_performance",
                         bash_command="cd /opt/project && /opt/venv/bin/python -m src.monitoring.drift")
    gate = ShortCircuitOperator(task_id="retrain_needed", python_callable=_should_retrain)
    retrain = TriggerDagRunOperator(task_id="trigger_training", trigger_dag_id="species_training",
                                    conf={"reason": "drift or accuracy drop"})
    check >> gate >> retrain
