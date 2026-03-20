"""SHAP sensitivity analysis and explainability for the trained XGBoost model.

Computes SHAP TreeExplainer values on the held-out test set, generates
global feature-importance plots, and derives per-row top-feature JSON
for downstream API consumption.

Artifacts produced:
  - Global feature importance bar plot (mean |SHAP|)
  - Beeswarm plot
  - Top-20 feature summary table (CSV)
  - SHAP values parquet (precomputed for dashboard / batch inference)
  - Per-row top_features_json column on scored output
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI/server environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from xgboost import XGBClassifier

from src.config.settings import settings
from src.models.train import (
    DEFAULT_MODEL_URI,
    DEFAULT_TEST_URI,
    DEFAULT_VAL_URI,
    LABEL_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default URIs / paths
# ---------------------------------------------------------------------------
DEFAULT_SHAP_PARQUET_URI = "gs://financial-distress-data/shap/shap_values.parquet"
DEFAULT_SHAP_ARTIFACT_DIR = "artifacts/shap"


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _get_shap() -> Any:
    """Import shap lazily to avoid hard dependency during unrelated tests."""
    import shap

    return shap


def _get_mlflow() -> Any:
    """Import mlflow lazily to avoid hard dependency during unrelated tests."""
    import mlflow

    return mlflow


# ---------------------------------------------------------------------------
# GCS helpers (reuse pattern from train.py / evaluate.py)
# ---------------------------------------------------------------------------


def _gcs_client() -> Any:
    """Create a Google Cloud Storage client."""
    from google.cloud import storage

    return storage.Client()


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse gs://bucket/path into (bucket, path)."""
    if not gcs_uri.startswith("gs://"):
        msg = f"Expected gs:// URI, got {gcs_uri}"
        raise ValueError(msg)
    stripped = gcs_uri.replace("gs://", "", 1)
    if "/" not in stripped:
        msg = f"Expected gs://bucket/path, got {gcs_uri}"
        raise ValueError(msg)
    bucket_name, blob_path = stripped.split("/", 1)
    return bucket_name, blob_path


def _upload_local_to_gcs(local_path: Path, gcs_uri: str) -> None:
    """Upload a local file to a GCS object path."""
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))
    log.info("Uploaded %s -> %s", local_path, gcs_uri)


def _download_gcs_to_local(gcs_uri: str, local_path: Path) -> None:
    """Download a GCS object to a local path."""
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))


# ---------------------------------------------------------------------------
# Model / data loading  (mirrors evaluate.py patterns)
# ---------------------------------------------------------------------------


def _load_split(split_uri: str) -> pd.DataFrame:
    """Load a parquet split from a local path or gs:// URI."""
    return pd.read_parquet(split_uri)


def _load_xgb_model(model_uri: str) -> XGBClassifier:
    """Load an XGBoost model artifact from GCS or local path."""
    model = XGBClassifier()
    if model_uri.startswith("gs://"):
        _, blob_path = _parse_gcs_uri(model_uri)
        suffix = Path(blob_path).suffix or ".model"
        local_model_path = Path(tempfile.gettempdir()) / f"best_xgb_model{suffix}"
        _download_gcs_to_local(model_uri, local_model_path)
        model.load_model(str(local_model_path))
        return model
    model.load_model(model_uri)
    return model


def _align_feature_frame(features: pd.DataFrame, trained_columns: list[str]) -> pd.DataFrame:
    """Align a feature frame to the trained model's feature column set."""
    out = features.copy()
    for col in out.columns:
        if is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype("int64")
    out = pd.get_dummies(out, dummy_na=True)
    return out.reindex(columns=list(trained_columns), fill_value=0)


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: XGBClassifier,
    feature_matrix: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP values using TreeExplainer.

    Args:
        model: Trained XGBClassifier.
        feature_matrix: Aligned numeric feature matrix (same columns as training).

    Returns:
        2-D numpy array of SHAP values, shape (n_samples, n_features).
        For binary classification the positive-class SHAP values are returned.
    """
    shap = _get_shap()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_matrix)

    # shap_values may be a list [class_0, class_1] for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    shap_arr: np.ndarray = np.asarray(shap_values)
    log.info(
        "Computed SHAP values: %d samples × %d features",
        shap_arr.shape[0],
        shap_arr.shape[1],
    )
    return shap_arr


# ---------------------------------------------------------------------------
# SHAP plots
# ---------------------------------------------------------------------------


def save_feature_importance_bar_plot(
    shap_values: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    top_n: int = 20,
) -> Path:
    """Generate and save a global feature importance bar plot (mean |SHAP|).

    Args:
        shap_values: 2-D array of SHAP values.
        feature_names: Feature column names matching shap_values columns.
        out_path: File path to save the PNG.
        top_n: Number of top features to display.

    Returns:
        Path to the saved plot.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{top_n} Global Feature Importance (mean |SHAP|)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved feature importance bar plot: %s", out_path)
    return out_path


def save_beeswarm_plot(
    shap_values: np.ndarray,
    feature_matrix: pd.DataFrame,
    out_path: Path,
    max_display: int = 20,
) -> Path:
    """Generate and save a SHAP beeswarm plot.

    Args:
        shap_values: 2-D array of SHAP values.
        feature_matrix: Original feature values for coloring.
        out_path: File path to save the PNG.
        max_display: Number of features to display.

    Returns:
        Path to the saved plot.
    """
    shap = _get_shap()
    explanation = shap.Explanation(
        values=shap_values,
        data=feature_matrix.values,
        feature_names=list(feature_matrix.columns),
    )

    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved beeswarm plot: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Top-20 summary table
# ---------------------------------------------------------------------------


def build_top_features_table(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Build a summary table of the top-N most important features.

    Columns: rank, feature, mean_abs_shap, mean_shap, std_shap, direction.

    Args:
        shap_values: 2-D array of SHAP values.
        feature_names: Feature column names.
        top_n: How many top features to include.

    Returns:
        DataFrame with one row per top feature.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_raw = shap_values.mean(axis=0)
    std_vals = shap_values.std(axis=0)

    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "mean_shap": mean_raw,
            "std_shap": std_vals,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    summary = summary.head(top_n).reset_index(drop=True)
    summary.index = summary.index + 1
    summary.index.name = "rank"

    # Direction: positive mean => increases distress risk, negative => protective
    summary["direction"] = np.where(summary["mean_shap"] > 0, "increases_risk", "protective")

    return summary


# ---------------------------------------------------------------------------
# Per-row top features JSON
# ---------------------------------------------------------------------------


def derive_top_features_json(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
) -> list[str]:
    """Derive a JSON string per row with the top-K SHAP contributors.

    Each JSON string has the format:
      [{"feature": "debt_to_equity", "shap_value": 0.12, "rank": 1}, ...]

    Args:
        shap_values: 2-D array of SHAP values.
        feature_names: Feature column names.
        top_k: Number of top contributors per row.

    Returns:
        List of JSON strings, one per sample row.
    """
    n_samples = shap_values.shape[0]
    result: list[str] = []

    for i in range(n_samples):
        row_vals = shap_values[i]
        abs_vals = np.abs(row_vals)
        top_indices = np.argsort(abs_vals)[::-1][:top_k]

        contributors = []
        for rank, idx in enumerate(top_indices, start=1):
            contributors.append(
                {
                    "feature": feature_names[idx],
                    "shap_value": round(float(row_vals[idx]), 6),
                    "rank": rank,
                }
            )

        result.append(json.dumps(contributors))

    log.info("Derived top_%d features JSON for %d rows", top_k, n_samples)
    return result


# ---------------------------------------------------------------------------
# SHAP values parquet serialization
# ---------------------------------------------------------------------------


def save_shap_parquet(
    shap_values: np.ndarray,
    feature_names: list[str],
    eval_df: pd.DataFrame,
    top_features_json: list[str],
    out_path: Path,
    gcs_uri: str | None = None,
) -> Path:
    """Save precomputed SHAP values as a parquet file.

    Includes identifier columns (firm_id, fiscal_year, fiscal_period if present)
    alongside SHAP values and the top_features_json column.

    Args:
        shap_values: 2-D array of SHAP values.
        feature_names: Feature column names.
        eval_df: Original evaluation DataFrame (for identifiers).
        top_features_json: Per-row JSON strings.
        out_path: Local file path.
        gcs_uri: Optional GCS URI to upload.

    Returns:
        Local path to the saved parquet.
    """
    # Build SHAP columns with shap_ prefix
    shap_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{f}" for f in feature_names],
        index=eval_df.index,
    )

    # Attach identifiers
    id_cols = ["firm_id", "fiscal_year", "fiscal_period"]
    for col in id_cols:
        if col in eval_df.columns:
            shap_df.insert(0, col, eval_df[col].values)

    shap_df["top_features_json"] = top_features_json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_parquet(out_path, index=False)
    log.info("Saved SHAP parquet: %s (%d rows)", out_path, len(shap_df))

    if gcs_uri:
        try:
            _upload_local_to_gcs(out_path, gcs_uri)
        except Exception:
            log.warning("GCS upload failed for SHAP parquet; local copy saved at %s", out_path)

    return out_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_shap_analysis(
    model_uri: str | None = None,
    test_uri: str | None = None,
    val_uri: str | None = None,
    shap_parquet_gcs_uri: str | None = None,
    output_dir: str | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Run the full SHAP analysis pipeline.

    Steps:
      1. Load model and test data
      2. Compute SHAP TreeExplainer values
      3. Generate plots (bar, beeswarm)
      4. Build top-20 feature summary table
      5. Derive per-row top_features_json
      6. Save SHAP parquet (local + optional GCS)
      7. Log all artifacts to MLflow

    Args:
        model_uri: Path or GCS URI to the trained XGBoost model.
        test_uri: Path or GCS URI to the test split parquet.
        val_uri: Path or GCS URI to the validation split parquet.
        shap_parquet_gcs_uri: GCS URI for uploading SHAP parquet.
        output_dir: Local directory for artifacts.
        log_to_mlflow: Whether to log artifacts to MLflow.

    Returns:
        Dict with paths to generated artifacts and summary stats.
    """
    resolved_model_uri = model_uri or os.getenv("MODEL_ARTIFACT_URI") or DEFAULT_MODEL_URI
    resolved_test_uri = test_uri or os.getenv("TEST_URI") or DEFAULT_TEST_URI
    resolved_val_uri = val_uri or os.getenv("VAL_URI") or DEFAULT_VAL_URI
    resolved_shap_gcs = (
        shap_parquet_gcs_uri or os.getenv("SHAP_PARQUET_URI") or DEFAULT_SHAP_PARQUET_URI
    )
    artifact_dir = Path(
        output_dir or os.getenv("SHAP_ARTIFACT_DIR") or DEFAULT_SHAP_ARTIFACT_DIR
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model and data ──────────────────────────────────────────
    log.info("Loading model from %s", resolved_model_uri)
    model = _load_xgb_model(resolved_model_uri)

    log.info("Loading test split from %s", resolved_test_uri)
    test_df = _load_split(resolved_test_uri)

    if LABEL_COL not in test_df.columns:
        msg = f"Missing label column in test data: {LABEL_COL}"
        raise ValueError(msg)

    eval_df = test_df.copy()
    eval_x_raw = eval_df.drop(columns=[LABEL_COL])

    trained_columns = model.get_booster().feature_names
    if not trained_columns:
        # Fallback: load val set to infer columns (same approach as evaluate.py)
        val_df = _load_split(resolved_val_uri)
        val_x = pd.get_dummies(val_df.drop(columns=[LABEL_COL]), dummy_na=True)
        trained_columns = list(val_x.columns)

    eval_x_aligned = _align_feature_frame(eval_x_raw, list(trained_columns))
    feature_names = list(eval_x_aligned.columns)

    # ── 2. Compute SHAP values ──────────────────────────────────────────
    log.info("Computing SHAP values via TreeExplainer...")
    shap_values = compute_shap_values(model, eval_x_aligned)

    # ── 3. Generate plots ───────────────────────────────────────────────
    bar_path = save_feature_importance_bar_plot(
        shap_values, feature_names, artifact_dir / "shap_feature_importance.png"
    )
    beeswarm_path = save_beeswarm_plot(
        shap_values, eval_x_aligned, artifact_dir / "shap_beeswarm.png"
    )

    # ── 4. Top-20 summary table ─────────────────────────────────────────
    top_table = build_top_features_table(shap_values, feature_names, top_n=20)
    top_table_path = artifact_dir / "shap_top20_features.csv"
    top_table.to_csv(top_table_path)
    log.info("Saved top-20 feature table: %s", top_table_path)

    # ── 5. Per-row top_features_json ────────────────────────────────────
    top_features_json = derive_top_features_json(shap_values, feature_names, top_k=3)

    # ── 6. Save SHAP parquet ────────────────────────────────────────────
    shap_parquet_path = save_shap_parquet(
        shap_values=shap_values,
        feature_names=feature_names,
        eval_df=eval_df,
        top_features_json=top_features_json,
        out_path=artifact_dir / "shap_values.parquet",
        gcs_uri=resolved_shap_gcs,
    )

    # ── 7. Log to MLflow ────────────────────────────────────────────────
    if log_to_mlflow:
        mlflow = _get_mlflow()
        if settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name="shap-analysis") as run:
            mlflow.log_param("model_uri", resolved_model_uri)
            mlflow.log_param("test_uri", resolved_test_uri)
            mlflow.log_param("n_samples", len(eval_df))
            mlflow.log_param("n_features", len(feature_names))

            # Log top-5 feature importances as metrics
            for _, row in top_table.head(5).iterrows():
                safe_name = str(row["feature"])[:200]
                mlflow.log_metric(f"shap_importance_{safe_name}", row["mean_abs_shap"])

            mlflow.log_artifact(str(bar_path), artifact_path="shap_plots")
            mlflow.log_artifact(str(beeswarm_path), artifact_path="shap_plots")
            mlflow.log_artifact(str(top_table_path), artifact_path="shap_tables")
            mlflow.log_artifact(str(shap_parquet_path), artifact_path="shap_data")

            log.info("Logged SHAP artifacts to MLflow run %s", run.info.run_id)
            mlflow_run_id = run.info.run_id
    else:
        mlflow_run_id = None

    result = {
        "bar_plot": str(bar_path),
        "beeswarm_plot": str(beeswarm_path),
        "top20_table": str(top_table_path),
        "shap_parquet": str(shap_parquet_path),
        "n_samples": len(eval_df),
        "n_features": len(feature_names),
        "top_feature": str(top_table.iloc[0]["feature"]) if len(top_table) > 0 else "",
        "mlflow_run_id": mlflow_run_id,
    }
    log.info("SHAP analysis complete: %s", json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for SHAP analysis."""
    result = run_shap_analysis()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
