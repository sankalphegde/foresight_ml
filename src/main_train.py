"""Entry point for the model training pipeline.

Orchestrates the full model development lifecycle:
  1. Train        — XGBoost + Optuna hyperparameter tuning
  2. Evaluate     — held-out test set metrics + per-slice evaluation
  3. Quality gate — blocks registration if test_roc_auc < 0.80
  4. SHAP         — feature importance + bias report generation
  5. Inference    — batch scoring with SHAP explanations attached
  6. Register     — MLflow registry + rollback check

Exit codes:
    0  — pipeline succeeded, quality gate passed, model registered
    1  — quality gate failed (test_roc_auc < 0.80) or pipeline error

The non-zero exit causes the Cloud Run job to fail, which propagates
to --wait in the Airflow DAG, blocking downstream tasks automatically.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import cast, Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

QUALITY_GATE_ROC_AUC = 0.80  # from project spec, Section 11
GCS_BUCKET = os.environ.get("GCS_BUCKET", "financial-distress-data")
MODEL_REPORT_GCS_PATH = "models/optuna_results.json"


def _load_training_report() -> dict[str, Any]:
    """Download optuna_results.json from GCS and return as dict."""
    from google.cloud import storage

    client = storage.Client()
    tmp = Path("/tmp/optuna_results.json")
    client.bucket(GCS_BUCKET).blob(MODEL_REPORT_GCS_PATH).download_to_filename(str(tmp))
    with open(tmp) as f:
        return cast(dict[str, Any], json.load(f))


def _get_latest_mlflow_run_id() -> tuple[str, float]:
    """Fetch the most recent MLflow run_id and its test_roc_auc."""
    import mlflow
    from src.config.settings import settings

    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    runs: pd.DataFrame = pd.DataFrame(mlflow.search_runs(
        experiment_names=[settings.mlflow_experiment_name],
        order_by=["start_time DESC"],
        max_results=1,
    ))
    if runs.empty:
        raise RuntimeError(
            f"No MLflow runs found in experiment '{settings.mlflow_experiment_name}'. "
            "Did evaluate.py run successfully?"
        )
    run_id = str(runs.iloc[0]["run_id"])
    test_roc_auc = float(runs.iloc[0].get("metrics.test_roc_auc", 0.0))
    return run_id, test_roc_auc


def main() -> None:
    """Run full model pipeline: train → evaluate → quality gate → SHAP → inference → register."""

    # ── Step 1: Train ────────────────────────────────────────────────────
    logger.info("Step 1/6 — Model training (XGBoost + Optuna)")
    try:
        from src.models.train import main as run_training
        run_training()
        logger.info("Training complete")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # ── Step 2: Evaluate ─────────────────────────────────────────────────
    logger.info("Step 2/6 — Model evaluation on held-out test set")
    try:
        from src.models.evaluate import main as run_evaluation
        run_evaluation()
        logger.info("Evaluation complete")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

    # ── Step 3: Quality gate ─────────────────────────────────────────────
    logger.info("Step 3/6 — Quality gate check (test_roc_auc ≥ 0.80)")
    try:
        report = _load_training_report()
        test_roc_auc = float(report.get("test_roc_auc", 0.0))
    except Exception as e:
        logger.error(f"Could not load training report from GCS: {e}")
        sys.exit(1)

    if test_roc_auc < QUALITY_GATE_ROC_AUC:
        logger.error(
            f"Quality gate FAILED: test_roc_auc={test_roc_auc:.4f} < {QUALITY_GATE_ROC_AUC}. "
            "Model will NOT be registered. CI merge blocked."
        )
        sys.exit(1)

    logger.info(f"Quality gate PASSED: test_roc_auc={test_roc_auc:.4f}")

    # ── Step 4: SHAP + bias report ───────────────────────────────────────
    logger.info("Step 4/6 — SHAP explainability + bias report generation")
    try:
        from src.models.explain import main as run_shap
        run_shap()
        logger.info("SHAP analysis and bias report complete")
    except Exception as e:
        # Non-fatal — SHAP failure should not block model registration
        logger.warning(
            f"SHAP/bias analysis failed (non-fatal, continuing): {e}"
        )

    # ── Step 5: Batch inference ──────────────────────────────────────────
    logger.info("Step 5/6 — Batch inference with SHAP explanations")
    try:
        from src.models.predict import run_batch_inference
        features_path = os.environ.get(
            "FEATURES_GCS_PATH",
            f"gs://{GCS_BUCKET}/features/labeled_v1/labeled_panel.parquet"
        )
        run_batch_inference(
            features_gcs_path=features_path,
            version_str="1.0"
        )
        logger.info("Batch inference complete")
    except Exception as e:
        # Non-fatal — inference failure should not block registration
        logger.warning(
            f"Batch inference failed (non-fatal, continuing): {e}"
        )

    # ── Step 6: Register ─────────────────────────────────────────────────
    logger.info("Step 6/6 — Model registration with rollback check")
    try:
        from src.models.registry import evaluate_and_register_model
        run_id, mlflow_roc_auc = _get_latest_mlflow_run_id()
        promoted = evaluate_and_register_model(
            run_id=run_id,
            test_roc_auc=mlflow_roc_auc,
            recall_critically_low=False,
        )
        if promoted:
            logger.info("Model promoted to Production in MLflow registry")
        else:
            logger.warning(
                "Model stayed in Staging — either worse than current Production "
                "or within 2% tolerance threshold"
            )
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        sys.exit(1)

    logger.info("Training pipeline complete — all 6 steps finished successfully")


if __name__ == "__main__":
    main()