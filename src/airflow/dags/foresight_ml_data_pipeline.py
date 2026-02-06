import os

# Import ingestion functions
import sys
from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

from src.ingestion.fred_job import main as fred_main
from src.ingestion.sec_job import main as sec_main


def run_fred_ingestion(**context: Any) -> None:
    """Run FRED data ingestion"""
    execution_date = context["ds"]
    os.environ["EXECUTION_DATE"] = execution_date
    os.environ["GCS_BUCKET"] = os.getenv("GCS_BUCKET", "")
    os.environ["FRED_API_KEY"] = os.getenv("FRED_API_KEY", "")

    print(f"Running FRED ingestion for {execution_date}")
    fred_main()


def run_sec_ingestion(**context: Any) -> None:
    """Run SEC data ingestion"""
    execution_date = context["ds"]
    os.environ["EXECUTION_DATE"] = execution_date
    os.environ["GCS_BUCKET"] = os.getenv("GCS_BUCKET", "")
    os.environ["SEC_USER_AGENT"] = os.getenv("SEC_USER_AGENT", "")

    print(f"Running SEC ingestion for {execution_date}")
    sec_main()


with DAG(
    dag_id="foresight_ingestion",
    start_date=datetime(2020, 1, 1),  # Start from 2020 (6 years of data)
    schedule="@daily",  # Run daily to collect all historical data quickly
    catchup=True,  # Enable backfill to get historical data
    max_active_runs=3,  # Limit concurrent runs
    tags=["foresight-ml", "ingestion"],
) as dag:
    fred_task = PythonOperator(
        task_id="run_fred_ingestion",
        python_callable=run_fred_ingestion,
        provide_context=True,
    )

    sec_task = PythonOperator(
        task_id="run_sec_ingestion",
        python_callable=run_sec_ingestion,
        provide_context=True,
    )
