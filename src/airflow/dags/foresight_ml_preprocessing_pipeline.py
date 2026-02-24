"""Airflow DAG for Foresight ML preprocessing pipeline.

This DAG orchestrates the preprocessing of SEC XBRL and FRED data:
1. Sync raw data from GCS to local worker
2. Run preprocessing (format, clean, deduplicate)
3. Upload interim data back to GCS

Runs weekly to process any new raw data collected by the ingestion DAG.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator

sys.path.insert(0, "/opt/airflow")

from src.data.preprocess import run_preprocessing  # noqa: E402

# Configuration
GCS_BUCKET = os.getenv("GCS_BUCKET", "financial-distress-data")
LOCAL_RAW_DIR = Path("/tmp/airflow_preprocess/raw")
LOCAL_INTERIM_DIR = Path("/tmp/airflow_preprocess/interim")


def prepare_directories(**context):
    """Create local directories for preprocessing."""
    LOCAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    (LOCAL_RAW_DIR / "sec_xbrl").mkdir(parents=True, exist_ok=True)
    (LOCAL_RAW_DIR / "fred").mkdir(parents=True, exist_ok=True)
    LOCAL_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directories: {LOCAL_RAW_DIR}, {LOCAL_INTERIM_DIR}")


def run_preprocess_task(**context):
    """Execute the preprocessing pipeline."""
    print("Starting preprocessing with paths:")
    print(f"  SEC raw: {LOCAL_RAW_DIR / 'sec_xbrl'}")
    print(f"  FRED raw: {LOCAL_RAW_DIR / 'fred'}")
    print(f"  Output: {LOCAL_INTERIM_DIR}")

    report = run_preprocessing(
        sec_raw_dir=LOCAL_RAW_DIR / "sec_xbrl",
        fred_raw_dir=LOCAL_RAW_DIR / "fred",
        out_dir=LOCAL_INTERIM_DIR,
        bucket=GCS_BUCKET,
        sec_gcs_out="interim/sec_xbrl_long.parquet",
        fred_gcs_out="interim/fred_timeseries.parquet",
        report_gcs_out="interim/preprocess_report.json",
    )

    # Push report to XCom for downstream tasks
    context["task_instance"].xcom_push(key="preprocess_report", value=report)
    print("Preprocessing complete!")
    print(f"Report: {report}")


def cleanup_task(**context):
    """Clean up local temporary files."""
    import shutil

    if LOCAL_RAW_DIR.exists():
        shutil.rmtree(LOCAL_RAW_DIR)
        print(f"Cleaned up {LOCAL_RAW_DIR}")

    if LOCAL_INTERIM_DIR.exists():
        shutil.rmtree(LOCAL_INTERIM_DIR)
        print(f"Cleaned up {LOCAL_INTERIM_DIR}")


# DAG definition
default_args = {
    "owner": "foresight-ml",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

with DAG(
    dag_id="foresight_preprocessing",
    default_args=default_args,
    description="Preprocess SEC and FRED raw data into interim format",
    schedule="@weekly",  # Run weekly after data ingestion
    start_date=datetime(2026, 2, 1),
    catchup=False,
    max_active_runs=1,
    tags=["foresight-ml", "preprocessing", "etl"],
) as dag:
    # Task 1: Prepare local directories
    prepare_dirs = PythonOperator(
        task_id="prepare_directories",
        python_callable=prepare_directories,
    )

    # Task 2: Sync SEC raw data from GCS
    # Note: GCSToLocalFilesystemOperator doesn't support directory sync well,
    # so we use BashOperator with gsutil
    sync_sec_raw = BashOperator(
        task_id="sync_sec_raw_data",
        bash_command=f"gsutil -m rsync -r gs://{GCS_BUCKET}/raw/sec_xbrl {LOCAL_RAW_DIR}/sec_xbrl",
    )

    # Task 3: Sync FRED raw data from GCS
    sync_fred_raw = BashOperator(
        task_id="sync_fred_raw_data",
        bash_command=f"gsutil -m rsync -r gs://{GCS_BUCKET}/raw/fred {LOCAL_RAW_DIR}/fred",
    )

    # Task 4: Run preprocessing
    preprocess = PythonOperator(
        task_id="run_preprocessing",
        python_callable=run_preprocess_task,
    )

    # Task 5: Upload SEC interim data
    upload_sec = LocalFilesystemToGCSOperator(
        task_id="upload_sec_interim",
        src=str(LOCAL_INTERIM_DIR / "sec_xbrl_long.parquet"),
        dst="interim/sec_xbrl_long.parquet",
        bucket=GCS_BUCKET,
    )

    # Task 6: Upload FRED interim data
    upload_fred = LocalFilesystemToGCSOperator(
        task_id="upload_fred_interim",
        src=str(LOCAL_INTERIM_DIR / "fred_timeseries.parquet"),
        dst="interim/fred_timeseries.parquet",
        bucket=GCS_BUCKET,
    )

    # Task 7: Upload preprocessing report
    upload_report = LocalFilesystemToGCSOperator(
        task_id="upload_report",
        src=str(LOCAL_INTERIM_DIR / "preprocess_report.json"),
        dst="interim/preprocess_report.json",
        bucket=GCS_BUCKET,
    )

    # Task 8: Cleanup temporary files
    cleanup = PythonOperator(
        task_id="cleanup_temp_files",
        python_callable=cleanup_task,
        trigger_rule="all_done",  # Run even if upstream tasks fail
    )

    # Task dependencies
    prepare_dirs >> [sync_sec_raw, sync_fred_raw]
    [sync_sec_raw, sync_fred_raw] >> preprocess
    preprocess >> [upload_sec, upload_fred, upload_report]
    [upload_sec, upload_fred, upload_report] >> cleanup
