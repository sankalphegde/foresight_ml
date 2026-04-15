"""Retrain trigger — reads and clears the retrain flag from GCS.

Used by the Airflow BranchPythonOperator to decide whether to
trigger the training DAG or skip retraining.
"""

from __future__ import annotations

import json
import os

from src.utils.logging import get_logger

logger = get_logger(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "financial-distress-data")
RETRAIN_FLAG_PATH = "monitoring/triggers/retrain_flag.json"


def check_retrain_flag() -> bool:
    """Check if retrain flag exists in GCS.

    Reads the flag, logs the reason, then deletes it so it does
    not re-trigger on the next DAG run.

    Returns:
        True if retrain should be triggered, False otherwise.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(RETRAIN_FLAG_PATH)

    if not blob.exists():
        logger.info("No retrain flag found — skipping retraining")
        return False

    try:
        flag = json.loads(blob.download_as_text())
        logger.warning(
            "Retrain flag found — triggered_at=%s reason=%s drifted_features=%s",
            flag.get("triggered_at"),
            flag.get("reason"),
            flag.get("drifted_features", []),
        )
        # Delete flag after reading so it doesn't re-trigger
        blob.delete()
        logger.info("Retrain flag deleted from GCS")
        return True
    except Exception as e:
        logger.error("Failed to read retrain flag: %s — skipping retraining", e)
        return False


def branch_on_retrain_flag(**context: object) -> str:
    """Airflow BranchPythonOperator callable.

    Returns the task_id to execute next based on whether
    the retrain flag is set.
    """
    should_retrain = check_retrain_flag()
    if should_retrain:
        logger.info("Branching to: trigger_training_dag")
        return "trigger_training_dag"
    logger.info("Branching to: skip_retraining")
    return "skip_retraining"
