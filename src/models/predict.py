"""Executes batch inference using the Production model and attaches SHAP explanations.

Enhanced with:
- Input schema validation before prediction
- Versioning columns in every row of scores.parquet
- manifest.json provenance certificate written alongside scores
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.models.inference_schema import (
    IDENTITY_COLUMNS,
    validate_inference_input,
    validate_inference_output,
)
from src.models.manifest_io import upload_manifest_to_gcs, write_manifest
from src.models.manifest_schema import ManifestSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_batch_inference(features_gcs_path: str, version_str: str = "1.0") -> None:
    """Load Production model, score data, attach SHAP explanations and versioning.

    Pipeline flow:
        1. Load model and fetch version metadata from MLflow Registry
        2. Validate input DataFrame against inference schema
        3. Generate distress probability scores
        4. Attach confidence intervals and SHAP explanations
        5. Inject versioning columns into every row
        6. Validate output DataFrame
        7. Write scores.parquet + manifest.json to GCS

    Args:
        features_gcs_path: GCS URI to the input features parquet file.
        version_str: Version string for the output path (e.g. "1.0").
    """
    start_time = time.time()

    model_name = "foresight_xgboost"
    model_uri = f"models:/{model_name}/Production"

    # --- Step 1: Load model and fetch version metadata ---
    logger.info(f"Loading Production model from {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    client = MlflowClient()
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        raise ValueError(
            f"No Production version found for '{model_name}'. "
            "Run registry.py to promote a model first."
        )
    mv = prod_versions[0]
    run = client.get_run(mv.run_id)
    trained_at = datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc)
    model_roc_auc = run.data.metrics.get("test_roc_auc", 0.0)

    logger.info(
        "Model metadata: version=v%s, run_id=%s, roc_auc=%.4f",
        mv.version, mv.run_id, model_roc_auc,
    )

    # --- Step 2: Load and validate input ---
    logger.info(f"Loading latest features from {features_gcs_path}")
    latest_features_df = pd.read_parquet(features_gcs_path)

    input_errors = validate_inference_input(latest_features_df)
    if input_errors:
        error_msg = "Input validation failed:\n" + "\n".join(
            f"  - {e}" for e in input_errors
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info("Input schema validation passed (%d rows)", len(latest_features_df))

    # --- Step 3: Generate predictions ---
    # Capture pre-dummy raw feature columns for manifest
    raw_feature_columns = [
        c for c in latest_features_df.columns if c not in IDENTITY_COLUMNS
    ]

    X_predict = latest_features_df.drop(columns=IDENTITY_COLUMNS)
    predictions = model.predict(X_predict)
    latest_features_df["distress_probability"] = predictions

    # --- Step 4: Confidence intervals ---
    latest_features_df["confidence_interval_lower"] = np.clip(predictions - 0.05, 0, 1)
    latest_features_df["confidence_interval_upper"] = np.clip(predictions + 0.05, 0, 1)

    # --- Step 5: Inject versioning columns ---
    scored_at = datetime.now(timezone.utc)
    latest_features_df["model_version"] = f"v{mv.version}"
    latest_features_df["mlflow_run_id"] = mv.run_id
    latest_features_df["trained_at"] = trained_at.isoformat()
    latest_features_df["scored_at"] = scored_at.isoformat()
    latest_features_df["model_roc_auc"] = model_roc_auc

    # Load precomputed SHAP values from Person 4
    shap_path = "gs://financial-distress-data/shap/shap_values.parquet"
    logger.info(f"Loading SHAP values from {shap_path}")
    shap_df = pd.read_parquet(shap_path)

    # Extract the necessary columns
    shap_subset = shap_df[["firm_id", "fiscal_year", "fiscal_period", "top_features_json"]]

    # Attach precomputed SHAP top_features_json to each scored row
    final_scored_df = pd.merge(
        latest_features_df,
        shap_subset,
        on=["firm_id", "fiscal_year", "fiscal_period"],
        how="left",
    )

    # --- Step 6: Validate output ---
    output_errors = validate_inference_output(final_scored_df)
    if output_errors:
        logger.warning(
            "Output validation warnings:\n%s",
            "\n".join(f"  - {e}" for e in output_errors),
        )

    # --- Step 7: Write scores + manifest to GCS ---
    output_path = f"gs://financial-distress-data/inference/scores_v{version_str}/scores.parquet"
    final_scored_df.to_parquet(output_path, index=False)
    logger.info(f"Successfully saved batch inference results to {output_path}")

    # Build and write manifest.json
    duration = round(time.time() - start_time, 2)
    manifest = ManifestSchema(
        model_name=model_name,
        model_version=f"v{mv.version}",
        mlflow_run_id=mv.run_id,
        trained_at=trained_at,
        scored_at=scored_at,
        roc_auc=model_roc_auc,
        prediction_horizon=1,
        features_used=raw_feature_columns,
        row_count=len(final_scored_df),
        gcs_scores_path=output_path,
        inference_duration_seconds=duration,
    )

    local_manifest = Path("/tmp/manifest.json")
    write_manifest(manifest, local_manifest)
    gcs_dir = f"gs://financial-distress-data/inference/scores_v{version_str}"
    upload_manifest_to_gcs(local_manifest, gcs_dir)

    logger.info("Batch inference complete in %.2fs", duration)


if __name__ == "__main__":
    run_batch_inference(
        "gs://financial-distress-data/features/latest.parquet", version_str="1.0"
    )
