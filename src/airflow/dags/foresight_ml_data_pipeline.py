"""Airflow DAG for Foresight ML data ingestion pipeline."""

# ruff: noqa: I001
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from google.cloud import bigquery, storage

sys.path.insert(0, "/opt/airflow")

from src.ingestion.fred_increment_job import main as fred_main  # noqa: E402
from src.ingestion.sec_xbrl_increment_job import main as sec_main  # noqa: E402
from src.main_labeling import main as label_main  # noqa: E402
from src.main_panel import main as panel_main  # noqa: E402


def run_fred_ingestion(**context: Any) -> None:
    """Run FRED data ingestion."""
    logical_date = context.get("logical_date")
    execution_date = context.get("ds") or (
        logical_date.date().isoformat() if logical_date else datetime.utcnow().date().isoformat()
    )
    os.environ["EXECUTION_DATE"] = execution_date
    os.environ["GCS_BUCKET"] = os.getenv("GCS_BUCKET", "")
    os.environ["FRED_API_KEY"] = os.getenv("FRED_API_KEY", "")

    print(f"Running FRED ingestion for {execution_date}")
    fred_main()


def run_sec_ingestion(**context: Any) -> None:
    """Run SEC data ingestion."""
    logical_date = context.get("logical_date")
    execution_date = context.get("ds") or (
        logical_date.date().isoformat() if logical_date else datetime.utcnow().date().isoformat()
    )
    os.environ["EXECUTION_DATE"] = execution_date
    os.environ["GCS_BUCKET"] = os.getenv("GCS_BUCKET", "")
    os.environ["SEC_USER_AGENT"] = os.getenv("SEC_USER_AGENT", "")

    print(f"Running SEC ingestion for {execution_date}")
    sec_main()


def run_preprocess_demo(**context: Any) -> None:
    """Demo-safe preprocess validation step after ingestion."""
    bucket_name = os.getenv("GCS_BUCKET", "")
    if not bucket_name:
        raise RuntimeError("Missing required environment variable: GCS_BUCKET")

    project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    client = storage.Client(project=project_id)

    sec_probe = list(client.list_blobs(bucket_name, prefix="raw/sec_xbrl/", max_results=1))
    fred_probe = list(client.list_blobs(bucket_name, prefix="raw/fred/", max_results=1))

    if not sec_probe:
        raise RuntimeError("No SEC raw data found at gs://{}/raw/sec_xbrl/".format(bucket_name))
    if not fred_probe:
        raise RuntimeError("No FRED raw data found at gs://{}/raw/fred/".format(bucket_name))

    print("Preprocess demo check passed: raw SEC/FRED data exists in GCS.")


def run_bigquery_cleaning(**context: Any) -> None:
    """Run BigQuery cleaning SQL from repository file."""
    project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "financial-distress-ew"
    bucket_name = os.getenv("GCS_BUCKET", "financial-distress-data")

    sql_path = Path("/opt/airflow/src/data/cleaned/data_cleaned.sql")
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8")
    sql_text = sql_text.replace("financial-distress-ew", project_id)
    sql_text = sql_text.replace("financial-distress-data", bucket_name)

    statements = [statement.strip() for statement in sql_text.split(";") if statement.strip()]
    if not statements:
        raise RuntimeError("No executable SQL statements found in data_cleaned.sql")

    client = bigquery.Client(project=project_id)
    for statement in statements:
        client.query(statement).result()

    print("BigQuery cleaning completed.")


def run_panel_build(**context: Any) -> None:
    """Run panel construction job."""
    panel_main()


def run_labeling(**context: Any) -> None:
    """Run labeling job."""
    label_main()


with DAG(
    dag_id="foresight_ingestion",
    schedule="@daily",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    max_active_runs=3,  # Limit concurrent runs
    tags=["foresight-ml", "ingestion"],
) as dag:
    fred_task = PythonOperator(
        task_id="run_fred_ingestion",
        python_callable=run_fred_ingestion,
    )

    sec_task = PythonOperator(
        task_id="run_sec_ingestion",
        python_callable=run_sec_ingestion,
    )

    preprocess_task = PythonOperator(
        task_id="run_preprocess_ingested_data",
        python_callable=run_preprocess_demo,
    )

    bigquery_clean_task = PythonOperator(
        task_id="run_bigquery_cleaning",
        python_callable=run_bigquery_cleaning,
    )

    panel_task = PythonOperator(
        task_id="run_panel_build",
        python_callable=run_panel_build,
    )

    labeling_task = PythonOperator(
        task_id="run_labeling",
        python_callable=run_labeling,
    )

    [fred_task, sec_task] >> preprocess_task >> bigquery_clean_task >> panel_task >> labeling_task
