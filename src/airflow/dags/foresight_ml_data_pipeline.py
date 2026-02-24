"""Airflow DAG for Foresight ML data ingestion pipeline."""

# ruff: noqa: I001
import os
import sys
from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

from src.ingestion.fred_increment_job import main as fred_main  # noqa: E402
from src.ingestion.sec_xbrl_increment_job import main as sec_main  # noqa: E402


def run_fred_ingestion(**context: Any) -> None:
    """Run FRED data ingestion."""
    execution_date = context["ds"]
    os.environ["EXECUTION_DATE"] = execution_date
    os.environ["GCS_BUCKET"] = os.getenv("GCS_BUCKET", "")
    os.environ["FRED_API_KEY"] = os.getenv("FRED_API_KEY", "")

    print(f"Running FRED ingestion for {execution_date}")
    fred_main()


def run_sec_ingestion(**context: Any) -> None:
    """Run SEC data ingestion."""
    execution_date = context["ds"]
    os.environ["EXECUTION_DATE"] = execution_date
    os.environ["GCS_BUCKET"] = os.getenv("GCS_BUCKET", "")
    os.environ["SEC_USER_AGENT"] = os.getenv("SEC_USER_AGENT", "")

    print(f"Running SEC ingestion for {execution_date}")
    sec_main()


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
