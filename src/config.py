"""Project configuration: params.yaml for ML settings, environment variables for infrastructure."""
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
METADATA_CSV = RAW_DIR / "metadata.csv"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'species.db'}")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "species-id")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL", "species-classifier")
CHAMPION_ALIAS = "champion"

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
API_URL = os.getenv("API_URL", "http://api:8000")
API_ADMIN_USER = os.getenv("API_ADMIN_USER", "admin")
API_ADMIN_PASSWORD = os.getenv("API_ADMIN_PASSWORD", "change-me")

# iNaturalist asks API clients to identify themselves.
USER_AGENT = os.getenv("INAT_USER_AGENT", "species-id-mlops-course-project/0.1")


@lru_cache
def params(path: str | None = None) -> dict:
    with open(path or ROOT / "params.yaml") as f:
        return yaml.safe_load(f)
