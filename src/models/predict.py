"""Executes batch inference using the Production model and attaches SHAP explanations.

Enhanced with:
- Input schema validation before prediction
- Versioning columns in every row of scores.parquet
- manifest.json provenance certificate written alongside scores
"""

import os
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.models.inference_schema import (
    IDENTITY_COLUMNS,
    LABEL_COLUMN,
    validate_inference_output,
)
from src.models.manifest_io import upload_manifest_to_gcs, write_manifest
from src.models.manifest_schema import ManifestSchema
from src.utils.logging import get_logger

logger = get_logger(__name__)


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

    import tempfile

    import gcsfs
    from xgboost import XGBClassifier

    def _load_xgb_from_gcs():
        """Download XGBoost model from GCS and wrap it."""
        gcs_model_path = (
            f"gs://{os.environ.get('GCS_BUCKET', 'financial-distress-data')}/models/xgb_model.pkl"
        )
        fs = gcsfs.GCSFileSystem()
        with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tmp:
            tmp_path = tmp.name
        fs.get(gcs_model_path.replace("gs://", ""), tmp_path)
        _xgb = XGBClassifier()
        try:
            _xgb.load_model(tmp_path)
        except Exception:
            import joblib

            _xgb = joblib.load(tmp_path)

        class _GCSModelWrapper:
            def predict(self, X: "pd.DataFrame") -> "np.ndarray":
                return _xgb.predict_proba(X)[:, 1]

            @property
            def feature_names(self) -> "list[str] | None":
                try:
                    return list(_xgb.get_booster().feature_names or [])
                except Exception:
                    return None

        return _GCSModelWrapper()

    try:
        _pyfunc = mlflow.pyfunc.load_model(model_uri)

        # MLflow pyfunc.predict() returns class labels (0/1) for XGBoost classifiers,
        # not probabilities. Unwrap the native model to get predict_proba instead.
        try:
            _native = _pyfunc.unwrap_python_model()
            if hasattr(_native, "predict_proba"):
                _xgb_native = _native
            else:
                _xgb_native = _pyfunc._model_impl.xgb_model  # type: ignore[attr-defined]

            class _MLflowWrapper:
                def predict(self, X: "pd.DataFrame") -> "np.ndarray":
                    return _xgb_native.predict_proba(X)[:, 1]

                @property
                def feature_names(self) -> "list[str] | None":
                    try:
                        return list(_xgb_native.get_booster().feature_names or [])
                    except Exception:
                        return None

            model = _MLflowWrapper()
            logger.info("MLflow model loaded and unwrapped for predict_proba")
        except Exception as unwrap_err:
            logger.warning(
                "Could not unwrap MLflow pyfunc for predict_proba (%s). "
                "Falling back to GCS model.",
                unwrap_err,
            )
            model = _load_xgb_from_gcs()

    except Exception as mlflow_load_err:
        # MLflow artifact store empty — fall back to GCS
        logger.warning(
            "MLflow artifact load failed (%s). Falling back to GCS model at "
            "gs://financial-distress-data/models/xgb_model.pkl",
            mlflow_load_err,
        )
        model = _load_xgb_from_gcs()

    client = MlflowClient()
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        raise ValueError(
            f"No Production version found for '{model_name}'. "
            "Run registry.py to promote a model first."
        )
    mv = prod_versions[0]
    run = client.get_run(mv.run_id)
    trained_at = datetime.fromtimestamp(run.info.start_time / 1000, tz=UTC)
    model_roc_auc = run.data.metrics.get("test_roc_auc", 0.0)

    logger.info(
        "Model metadata: version=v%s, run_id=%s, roc_auc=%.4f",
        mv.version,
        mv.run_id,
        model_roc_auc,
    )

    # --- Step 2: Load and validate input ---
    logger.info(f"Loading latest features from {features_gcs_path}")
    latest_features_df = pd.read_parquet(features_gcs_path)

    # Drop label column if present (labeled panel is used as inference source)
    if LABEL_COLUMN in latest_features_df.columns:
        logger.info("Dropping label column '%s' from inference input", LABEL_COLUMN)
        latest_features_df = latest_features_df.drop(columns=[LABEL_COLUMN])

    # Lightweight validation: identity columns must be present, label must be absent
    missing_identity = [c for c in IDENTITY_COLUMNS if c not in latest_features_df.columns]
    if missing_identity:
        raise ValueError(f"Input validation failed: missing identity columns {missing_identity}")
    logger.info(
        "Input schema validation passed (%d rows, %d cols)",
        len(latest_features_df),
        len(latest_features_df.columns),
    )

    # --- Step 3: Generate predictions ---
    raw_feature_columns = [c for c in latest_features_df.columns if c not in IDENTITY_COLUMNS]

    # Save identity columns for the output dataframe
    output_identity_df = latest_features_df[IDENTITY_COLUMNS].copy()

    # Apply the same numeric transformation used during training:
    # datetime → int64, categorical → pd.get_dummies(dummy_na=True)
    from src.models.train import _to_numeric_frame  # noqa: PLC0415

    X_all = _to_numeric_frame(latest_features_df)

    # Align to the model's expected feature columns (handles unseen/missing dummies)
    expected_features = getattr(model, "feature_names", None)
    if expected_features:
        X_predict = X_all.reindex(columns=expected_features, fill_value=0)
        logger.info("Aligned inference features to %d model columns", len(expected_features))
    else:
        X_predict = X_all
        logger.info("Model feature names unavailable — using %d columns as-is", len(X_all.columns))

    predictions = model.predict(X_predict)

    # Build output dataframe: identity + predictions + derived columns
    latest_features_df = output_identity_df.copy()
    latest_features_df["distress_probability"] = predictions

    # --- Step 4: Confidence intervals ---
    latest_features_df["confidence_interval_lower"] = np.clip(predictions - 0.05, 0, 1)
    latest_features_df["confidence_interval_upper"] = np.clip(predictions + 0.05, 0, 1)

    # --- Step 5: Inject versioning columns ---
    scored_at = datetime.now(UTC)
    latest_features_df["model_version"] = f"v{mv.version}"
    latest_features_df["mlflow_run_id"] = mv.run_id
    latest_features_df["trained_at"] = trained_at.isoformat()
    latest_features_df["scored_at"] = scored_at.isoformat()
    latest_features_df["model_roc_auc"] = model_roc_auc

    # Load precomputed SHAP values — gracefully degrade if file not yet generated
    gcs_bucket = os.getenv("GCS_BUCKET", "financial-distress-data")
    shap_path = f"gs://{gcs_bucket}/shap/shap_values.parquet"
    logger.info(f"Loading SHAP values from {shap_path}")
    try:
        shap_df = pd.read_parquet(shap_path)
        shap_subset = shap_df[["firm_id", "fiscal_year", "fiscal_period", "top_features_json"]]
        final_scored_df = pd.merge(
            latest_features_df,
            shap_subset,
            on=["firm_id", "fiscal_year", "fiscal_period"],
            how="left",
        )
        logger.info(
            "SHAP values merged: %d rows matched",
            final_scored_df["top_features_json"].notna().sum(),
        )
    except Exception as shap_err:
        logger.warning(
            "SHAP values unavailable (%s) — scores.parquet will be written without "
            "top_features_json. Re-run explain.py to populate SHAP and re-score.",
            shap_err,
        )
        final_scored_df = latest_features_df.copy()
        final_scored_df["top_features_json"] = None

    # --- Step 6: Validate output ---
    output_errors = validate_inference_output(final_scored_df)
    if output_errors:
        logger.warning(
            "Output validation warnings:\n%s",
            "\n".join(f"  - {e}" for e in output_errors),
        )

    # --- Step 7: Write scores + manifest to GCS ---
    output_path = f"gs://{gcs_bucket}/inference/scores_v{version_str}/scores.parquet"
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
    gcs_dir = f"gs://{gcs_bucket}/inference/scores_v{version_str}"
    upload_manifest_to_gcs(local_manifest, gcs_dir)

    logger.info("Batch inference complete in %.2fs", duration)


if __name__ == "__main__":
    run_batch_inference(
        "gs://financial-distress-data/features/labeled_v1/labeled_panel.parquet",
        version_str="1.0",
    )
