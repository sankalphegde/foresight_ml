"""Airflow DAG for Foresight-ML data pipeline."""
import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import (
    LocalFilesystemToGCSOperator,
)

# Add src to path
sys.path.insert(0, "/opt/airflow/dags/src")
from src.data.clients.fred_client import FREDClient
from src.data.clients.sec_client import SECClient

# Configuration
GCS_BUCKET = os.getenv("GCS_BUCKET", "foresight-ml-data")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
FRED_API_KEY = os.getenv("FRED_API_KEY")

COMPANIES = [
    {"ticker": "AAPL", "cik": "0000320193"},
    {"ticker": "MSFT", "cik": "0000789019"},
    {"ticker": "GOOGL", "cik": "0001652044"},
    {"ticker": "AMZN", "cik": "0001018724"},
]


def fetch_sec_data(**context) -> str:  # type: ignore[no-untyped-def]
    """Fetch SEC filings for all companies."""
    if not SEC_USER_AGENT:
        raise ValueError("SEC_USER_AGENT environment variable not set")

    client = SECClient(user_agent=SEC_USER_AGENT, cache_dir="/tmp/cache/sec")

    all_filings = []
    for company in COMPANIES:
        filings_data = client.get_company_filings(company["cik"])
        filings = client.filter_filings(
            filings_data, form_types=["10-K", "10-Q"], start_date="2020-01-01"
        )

        # Convert Pydantic models to dicts
        for filing in filings:
            filing_dict = filing.model_dump()
            filing_dict["ticker"] = company["ticker"]
            all_filings.append(filing_dict)

    # Save to file
    output_path = "/tmp/sec_filings.json"
    with open(output_path, "w") as f:
        json.dump(all_filings, f)

    print(f"Fetched {len(all_filings)} SEC filings")
    return output_path


def fetch_fred_data(**context) -> str:  # type: ignore[no-untyped-def]
    """Fetch FRED economic indicators."""
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY environment variable not set")

    client = FREDClient(api_key=FRED_API_KEY, cache_dir="/tmp/cache/fred")

    df = client.get_common_indicators(start_date="2020-01-01", frequency="q")

    # Save to CSV
    output_path = "/tmp/fred_indicators.csv"
    df.to_csv(output_path)

    print(f"Fetched {len(df)} quarters of economic data")
    return output_path


def merge_data(**context) -> str:  # type: ignore[no-untyped-def]
    """Merge SEC and FRED data."""
    ti = context["ti"]
    sec_path = ti.xcom_pull(task_ids="fetch_sec_data")
    fred_path = ti.xcom_pull(task_ids="fetch_fred_data")

    # Load data
    with open(sec_path) as f:
        sec_data = json.load(f)
    fred_df = pd.read_csv(fred_path, index_col=0, parse_dates=True)

    # Convert SEC to DataFrame
    sec_df = pd.DataFrame(sec_data)
    sec_df["filing_date"] = pd.to_datetime(sec_df["filing_date"])
    sec_df["quarter"] = sec_df["filing_date"].dt.to_period("Q")

    # Merge with FRED (by quarter)
    fred_df["quarter"] = fred_df.index.to_period("Q")
    merged = sec_df.merge(fred_df.reset_index(), on="quarter", how="left")

    # Save merged data
    output_path = "/tmp/merged_data.parquet"
    merged.to_parquet(output_path)

    print(f"Merged data: {len(merged)} rows")
    return output_path

# DAG definition
default_args = {
    "owner": "foresight-ml",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "foresight_ml_data_pipeline",
    default_args=default_args,
    description="Fetch and process financial distress data",
    schedule_interval="@weekly",  # Run weekly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["foresight-ml", "data-pipeline"],
) as dag:
    # Task 1: Fetch SEC data
    fetch_sec = PythonOperator(
        task_id="fetch_sec_data",
        python_callable=fetch_sec_data,
    )

    # Task 2: Fetch FRED data
    fetch_fred = PythonOperator(
        task_id="fetch_fred_data",
        python_callable=fetch_fred_data,
    )

    # Task 3: Merge data
    merge = PythonOperator(
        task_id="merge_data",
        python_callable=merge_data,
    )

    # Task 4: Upload to GCS
    upload_to_gcs = LocalFilesystemToGCSOperator(
        task_id="upload_to_gcs",
        src="/tmp/merged_data.parquet",
        dst="raw/{{ ds }}/merged_data.parquet",
        bucket=GCS_BUCKET,
    )

    # Define task dependencies
    [fetch_sec, fetch_fred] >> merge >> upload_to_gcs