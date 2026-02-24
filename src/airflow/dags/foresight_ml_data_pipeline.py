"""Airflow DAG for Foresight ML data ingestion pipeline."""

# ruff: noqa: I001
import os
import subprocess
import sys
import tempfile
import json
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
from src.config.settings import settings  # noqa: E402
from src.data.validate_anomalies import validate_and_detect, upload_to_gcs  # noqa: E402
from src.utils.gcs import read_parquet_from_gcs  # noqa: E402


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


def run_validation_anomaly(**context: Any) -> None:
    """Run validation/anomaly checks on labeled panel and upload outputs to GCS."""
    bucket_name = settings.gcs_bucket or os.getenv("GCS_BUCKET", "")
    if not bucket_name:
        raise RuntimeError("Missing required environment variable: GCS_BUCKET")

    labeled_uri = f"gs://{bucket_name}/{settings.labeled_output_path}"
    df = read_parquet_from_gcs([labeled_uri])

    if "filed_date" in df.columns and "filing_date" not in df.columns:
        df = df.rename(columns={"filed_date": "filing_date"})

    anomalies, report = validate_and_detect(df)

    report_gcs_path = os.getenv("GCS_VALIDATION_REPORT_OUT", "processed/validation_report.json")
    anomalies_gcs_path = os.getenv("GCS_ANOMALIES_OUT", "processed/anomalies.parquet")

    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = Path(temp_dir) / "validation_report.json"
        anomalies_path = Path(temp_dir) / "anomalies.parquet"

        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
        anomalies.to_parquet(anomalies_path, index=False)

        upload_to_gcs(report_path, bucket_name, report_gcs_path)
        upload_to_gcs(anomalies_path, bucket_name, anomalies_gcs_path)

    print(f"Validation status: {report['status']}")
    print(f"Validation anomaly rows: {report['anomaly_count']}")

    fail_on_status = os.getenv("VALIDATION_FAIL_ON_STATUS", "false").lower() == "true"
    if fail_on_status and report["status"] != "pass":
        raise RuntimeError("Validation status failed. Check validation_report.json in GCS.")


def run_feature_bias_pipeline(**context: Any) -> None:
    """Run feature engineering + bias analysis pipeline in BigQuery mode."""
    pipeline_root = Path("/opt/airflow/src/feature_engineering")
    config_path = pipeline_root / "config" / "settings.yaml"

    if not pipeline_root.exists():
        raise FileNotFoundError(f"Feature engineering root not found: {pipeline_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"Feature engineering config not found: {config_path}")

    cmd = [
        sys.executable,
        "-m",
        "pipelines.run_pipeline",
        "--mode",
        "bigquery",
        "--config",
        str(config_path),
    ]

    subprocess_env = os.environ.copy()
    feature_bias_mode = os.getenv("FEATURE_BIAS_MODE", "safe").strip().lower()
    skip_heavy_visualizations = feature_bias_mode != "full"
    subprocess_env["SKIP_HEAVY_VISUALIZATIONS"] = "true" if skip_heavy_visualizations else "false"
    print(
        "Feature/Bias mode:",
        feature_bias_mode,
        "| SKIP_HEAVY_VISUALIZATIONS=",
        subprocess_env["SKIP_HEAVY_VISUALIZATIONS"],
    )

    completed = subprocess.run(
        cmd,
        cwd=str(pipeline_root),
        env=subprocess_env,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Feature/bias pipeline failed with exit code {completed.returncode}"
        )


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

    validation_anomaly_task = PythonOperator(
        task_id="run_validation_anomaly",
        python_callable=run_validation_anomaly,
    )

    feature_bias_task = PythonOperator(
        task_id="run_feature_bias_pipeline",
        python_callable=run_feature_bias_pipeline,
    )

    [fred_task, sec_task] >> preprocess_task >> bigquery_clean_task >> panel_task >> labeling_task >> feature_bias_task >> validation_anomaly_task
