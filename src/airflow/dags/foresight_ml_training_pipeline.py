"""Foresight ML Training Pipeline DAG.

Separate from the data ingestion DAG (foresight_ingestion).
Runs weekly - after data pipeline has produced labeled features.

Responsibility split:
- model_training.yml (CD):  deploys the updated training image to Cloud Run
- This DAG:                 executes the training job on schedule and checks
                            quality gate

Task order:
    check_data_ready → run_model_training → model_quality_gate
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "financial-distress-ew")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "financial-distress-data")

QUALITY_GATE_ROC_AUC = 0.80
LABELED_DATA_PATH = "features/labeled_v1/labeled_panel.parquet"
MODEL_REPORT_PATH = "models/optuna_results.json"
CLOUD_RUN_JOB_NAME = "foresight-training"


def _check_data_ready(**context: Any) -> None:
    """Gate: confirm labeled panel exists in GCS before starting training.

    Mirrors run_preprocess_demo in foresight_ml_data_pipeline.py —
    never let an expensive job start with missing inputs.
    """
    from google.cloud.storage import Client

    client = Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)

    if not bucket.blob(LABELED_DATA_PATH).exists():
        raise FileNotFoundError(
            f"Labeled panel not found at gs://{GCS_BUCKET}/{LABELED_DATA_PATH}. "
            "Run the data pipeline DAG (foresight_ingestion) first."
        )

    print(f"Data gate passed: gs://{GCS_BUCKET}/{LABELED_DATA_PATH} exists.")


def _run_model_training(**context: Any) -> None:
    """Execute the foresight-training Cloud Run job and wait for completion.

    Uses Google Cloud Run v2 Python client instead of gcloud CLI
    since gcloud is not available in the Airflow container.
    """
    from google.cloud import run_v2

    client = run_v2.JobsClient()
    job_name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/jobs/{CLOUD_RUN_JOB_NAME}"

    print(f"Triggering Cloud Run job: {job_name}")

    # Start the job execution
    request = run_v2.RunJobRequest(name=job_name)
    operation = client.run_job(request=request)

    print("Waiting for training job to complete (this may take 20-30 minutes)...")

    # .result() blocks until the operation completes
    # raises google.api_core.exceptions.GoogleAPICallError on failure
    result = operation.result()

    print(f"Training job completed successfully: {result.name if result else 'done'}")


def _model_quality_gate(**context: Any) -> None:
    """Read test_roc_auc from optuna_results.json written by train.py.

    Fails the DAG task if below threshold.
    Pushes metrics to XCom for visibility in Airflow UI.
    """
    from google.cloud.storage import Client

    client = Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(MODEL_REPORT_PATH)

    if not blob.exists():
        raise FileNotFoundError(
            f"Model report not found at gs://{GCS_BUCKET}/{MODEL_REPORT_PATH}. "
            "Training job may have failed before writing artifacts."
        )

    tmp = Path("/tmp/optuna_results.json")
    blob.download_to_filename(str(tmp))
    with open(tmp) as f:
        report = json.load(f)

    test_roc_auc = float(report.get("test_roc_auc", 0.0))

    if test_roc_auc < QUALITY_GATE_ROC_AUC:
        raise ValueError(
            f"Quality gate FAILED: test_roc_auc={test_roc_auc:.4f} "
            f"< {QUALITY_GATE_ROC_AUC}. Model NOT promoted to Production."
        )

    print(f"Quality gate PASSED: test_roc_auc={test_roc_auc:.4f}")

    ti = context["ti"]
    ti.xcom_push(key="test_roc_auc", value=test_roc_auc)
    ti.xcom_push(key="baseline_val_roc", value=report.get("baseline_val_roc"))
    ti.xcom_push(key="best_params", value=report.get("best_params", {}))


with DAG(
    dag_id="foresight_training",
    description="Weekly model training: data gate → train → evaluate → quality gate → register",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["foresight-ml", "training", "xgboost"],
    default_args={
        "retries": 1,
        "retry_delay": 300,
    },
) as dag:

    check_data_ready = PythonOperator(
        task_id="check_data_ready",
        python_callable=_check_data_ready,
    )

    run_model_training = PythonOperator(
        task_id="run_model_training",
        python_callable=_run_model_training,
        execution_timeout=None,
    )

    model_quality_gate = PythonOperator(
        task_id="model_quality_gate",
        python_callable=_model_quality_gate,
    )

    check_data_ready >> run_model_training >> model_quality_gate