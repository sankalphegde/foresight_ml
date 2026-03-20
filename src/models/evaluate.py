"""Model evaluation for held-out test data with MLflow tracking."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from src.config.settings import settings
from src.feature_engineering.pipelines.bias_analysis import create_slices
from src.models.train import DEFAULT_MODEL_URI, DEFAULT_TEST_URI, DEFAULT_VAL_URI, LABEL_COL

TOP_K_FRACTION = float(os.getenv("EVAL_TOP_K_FRACTION", "0.05"))
EVAL_START_YEAR = int(os.getenv("EVAL_START_YEAR", "2022"))
EVAL_END_YEAR = int(os.getenv("EVAL_END_YEAR", "2023"))


@dataclass(frozen=True)
class EvalMetrics:
    """Container for model evaluation metrics."""

    roc_auc: float
    precision_at_k: float
    recall_at_k: float
    brier_score: float
    f1_at_threshold: float
    sample_count: int
    positives: int


def _get_mlflow() -> Any:
    """Import mlflow lazily to avoid hard dependency during unrelated tests."""
    import mlflow

    return mlflow


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


def _download_gcs_to_local(gcs_uri: str, local_path: Path) -> None:
    """Download a GCS object to a local path."""
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))


def _load_split(split_uri: str) -> pd.DataFrame:
    """Load a parquet split from a local path or `gs://` URI."""
    return pd.read_parquet(split_uri)


def _align_feature_frame(features: pd.DataFrame, trained_columns: Sequence[str]) -> pd.DataFrame:
    """Align a feature frame to the trained model's feature column set."""
    out = features.copy()
    for column in out.columns:
        if is_datetime64_any_dtype(out[column]):
            out[column] = out[column].astype("int64")
    out = pd.get_dummies(out, dummy_na=True)
    return out.reindex(columns=list(trained_columns), fill_value=0)


def _extract_eval_window(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Filter dataframe to the held-out evaluation window when a date column exists."""
    if "fiscal_year" in df.columns:
        years = pd.to_numeric(df["fiscal_year"], errors="coerce")
        return df[(years >= start_year) & (years <= end_year)].copy()

    for date_col in ("filing_date", "filed_date"):
        if date_col in df.columns:
            years = pd.to_datetime(df[date_col], errors="coerce", utc=False).dt.year
            return df[(years >= start_year) & (years <= end_year)].copy()

    return df.copy()


def _precision_recall_at_k(
    y_true: pd.Series, y_score: np.ndarray, top_k_fraction: float
) -> tuple[float, float]:
    """Compute precision@k and recall@k for top risk-score observations."""
    if len(y_true) == 0:
        return float("nan"), float("nan")

    k = max(1, int(np.ceil(len(y_true) * top_k_fraction)))
    top_idx = np.argsort(y_score)[::-1][:k]
    top_true = y_true.iloc[top_idx]

    true_positives = int((top_true == 1).sum())
    predicted_positives = len(top_true)
    actual_positives = int((y_true == 1).sum())

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    return float(precision), float(recall)


def _safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    """Compute ROC-AUC, returning NaN when class coverage is insufficient."""
    if y_true.nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _tune_threshold_by_f1(y_true: pd.Series, y_score: np.ndarray) -> float:
    """Tune a decision threshold by maximizing F1 on a validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5

    denom = precision[:-1] + recall[:-1]
    f1_values = np.where(denom > 0, 2 * precision[:-1] * recall[:-1] / denom, 0.0)
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx])


def _compute_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    top_k_fraction: float,
) -> EvalMetrics:
    """Compute the full evaluation metric set for a scored dataset."""
    y_pred = (y_score >= threshold).astype(int)
    precision_at_k, recall_at_k = _precision_recall_at_k(y_true, y_score, top_k_fraction)

    return EvalMetrics(
        roc_auc=_safe_roc_auc(y_true, y_score),
        precision_at_k=float(precision_at_k),
        recall_at_k=float(recall_at_k),
        brier_score=float(brier_score_loss(y_true, y_score)),
        f1_at_threshold=float(f1_score(y_true, y_pred, zero_division=0)),
        sample_count=int(len(y_true)),
        positives=int((y_true == 1).sum()),
    )


def _save_global_plots(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    out_dir: Path,
) -> dict[str, Path]:
    """Generate and save ROC, PR, and confusion-matrix plots as PNG files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    roc_path = out_dir / "roc_curve.png"
    pr_path = out_dir / "precision_recall_curve.png"
    cm_path = out_dir / "confusion_matrix.png"

    if y_true.nunique(dropna=True) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150)
        plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()

    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix @ threshold={threshold:.3f}")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    return {
        "roc_curve": roc_path,
        "precision_recall_curve": pr_path,
        "confusion_matrix": cm_path,
    }


def _build_slice_performance_table(
    eval_df: pd.DataFrame,
    y_score: np.ndarray,
    threshold: float,
    top_k_fraction: float,
) -> pd.DataFrame:
    """Compute full metric set for each configured slice from bias analysis definitions."""
    slices = create_slices(eval_df)
    rows: list[dict[str, Any]] = []

    for dimension, slice_dict in slices.items():
        for slice_name, slice_df in slice_dict.items():
            if slice_df.empty or LABEL_COL not in slice_df.columns:
                continue

            slice_idx = slice_df.index
            slice_scores = pd.Series(y_score, index=eval_df.index).loc[slice_idx].to_numpy()
            slice_labels = slice_df[LABEL_COL]

            metrics = _compute_metrics(slice_labels, slice_scores, threshold, top_k_fraction)
            rows.append(
                {
                    "dimension": dimension,
                    "slice": str(slice_name),
                    "sample_count": metrics.sample_count,
                    "positives": metrics.positives,
                    "roc_auc": metrics.roc_auc,
                    "precision_at_5pct": metrics.precision_at_k,
                    "recall_at_5pct": metrics.recall_at_k,
                    "brier_score": metrics.brier_score,
                    "f1_at_tuned_threshold": metrics.f1_at_threshold,
                    "tuned_threshold": threshold,
                }
            )

    return pd.DataFrame(rows)


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


def export_run_comparison_for_notebook(
    output_csv_path: str = "artifacts/evaluation/mlflow_run_comparison.csv",
    experiment_name: str | None = None,
) -> str:
    """Export MLflow runs to CSV for `model_experiments.ipynb` comparison visuals."""
    mlflow = _get_mlflow()
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    resolved_experiment_name = experiment_name or settings.mlflow_experiment_name
    experiment = mlflow.get_experiment_by_name(resolved_experiment_name)
    if experiment is None:
        msg = f"MLflow experiment not found: {resolved_experiment_name}"
        raise ValueError(msg)

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_roc_auc DESC"],
    )
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(output_path, index=False)
    return str(output_path)


def evaluate_model(
    run_id: str | None = None, model_uri: str | None = None
) -> dict[str, str | float]:
    """Run held-out evaluation and log scalar metrics, plots, and per-slice performance to MLflow."""
    resolved_model_uri = model_uri or os.getenv("MODEL_ARTIFACT_URI") or DEFAULT_MODEL_URI
    val_uri = os.getenv("VAL_URI", DEFAULT_VAL_URI)
    test_uri = os.getenv("TEST_URI", DEFAULT_TEST_URI)

    val_df = _load_split(val_uri)
    test_df = _load_split(test_uri)

    eval_df = _extract_eval_window(test_df, EVAL_START_YEAR, EVAL_END_YEAR)
    if eval_df.empty:
        msg = (
            f"No rows found in evaluation window {EVAL_START_YEAR}-{EVAL_END_YEAR}. "
            "Check test split time coverage."
        )
        raise ValueError(msg)
    if LABEL_COL not in eval_df.columns:
        msg = f"Missing label column in eval data: {LABEL_COL}"
        raise ValueError(msg)
    if LABEL_COL not in val_df.columns:
        msg = f"Missing label column in validation data: {LABEL_COL}"
        raise ValueError(msg)

    model = _load_xgb_model(resolved_model_uri)

    val_x = pd.get_dummies(val_df.drop(columns=[LABEL_COL]), dummy_na=True)
    eval_x_raw = eval_df.drop(columns=[LABEL_COL])

    trained_columns = model.get_booster().feature_names
    if not trained_columns:
        trained_columns = list(val_x.columns)

    val_x_aligned = _align_feature_frame(val_x, trained_columns)
    eval_x_aligned = _align_feature_frame(eval_x_raw, trained_columns)

    val_y = val_df[LABEL_COL]
    eval_y = eval_df[LABEL_COL]

    val_scores = model.predict_proba(val_x_aligned)[:, 1]
    threshold = _tune_threshold_by_f1(val_y, val_scores)
    eval_scores = model.predict_proba(eval_x_aligned)[:, 1]

    metrics = _compute_metrics(eval_y, eval_scores, threshold, TOP_K_FRACTION)
    slice_table = _build_slice_performance_table(eval_df, eval_scores, threshold, TOP_K_FRACTION)

    mlflow = _get_mlflow()
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    run_kwargs: dict[str, Any] = {"run_name": "heldout-evaluation"}
    if run_id:
        run_kwargs = {"run_id": run_id}

    with mlflow.start_run(**run_kwargs) as run:
        mlflow.log_param("evaluated_model_uri", resolved_model_uri)
        mlflow.log_param("eval_test_uri", test_uri)
        mlflow.log_param("eval_year_start", EVAL_START_YEAR)
        mlflow.log_param("eval_year_end", EVAL_END_YEAR)
        mlflow.log_param("top_k_fraction", TOP_K_FRACTION)
        mlflow.log_param("tuned_threshold", threshold)

        mlflow.log_metric("test_roc_auc", metrics.roc_auc)
        mlflow.log_metric("test_precision_at_5pct", metrics.precision_at_k)
        mlflow.log_metric("test_recall_at_5pct", metrics.recall_at_k)
        mlflow.log_metric("test_brier_score", metrics.brier_score)
        mlflow.log_metric("test_f1_at_tuned_threshold", metrics.f1_at_threshold)
        mlflow.log_metric("test_sample_count", float(metrics.sample_count))
        mlflow.log_metric("test_positive_count", float(metrics.positives))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            plot_paths = _save_global_plots(eval_y, eval_scores, threshold, artifact_dir)
            slice_csv = artifact_dir / "slice_performance.csv"
            slice_json = artifact_dir / "slice_performance.json"
            summary_json = artifact_dir / "evaluation_summary.json"

            slice_table.to_csv(slice_csv, index=False)
            slice_table.to_json(slice_json, orient="records", indent=2)
            summary = {
                "model_uri": resolved_model_uri,
                "eval_window": [EVAL_START_YEAR, EVAL_END_YEAR],
                "metrics": {
                    "roc_auc": metrics.roc_auc,
                    "precision_at_5pct": metrics.precision_at_k,
                    "recall_at_5pct": metrics.recall_at_k,
                    "brier_score": metrics.brier_score,
                    "f1_at_tuned_threshold": metrics.f1_at_threshold,
                },
                "threshold": threshold,
                "slice_rows": int(len(slice_table)),
            }
            summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            for path in plot_paths.values():
                mlflow.log_artifact(str(path), artifact_path="evaluation_plots")
            mlflow.log_artifact(str(slice_csv), artifact_path="slice_metrics")
            mlflow.log_artifact(str(slice_json), artifact_path="slice_metrics")
            mlflow.log_artifact(str(summary_json), artifact_path="evaluation")

        notebook_path = export_run_comparison_for_notebook(
            output_csv_path="artifacts/evaluation/mlflow_run_comparison.csv"
        )
        mlflow.log_artifact(notebook_path, artifact_path="notebook_inputs")

        return {
            "evaluation_run_id": run.info.run_id,
            "model_uri": resolved_model_uri,
            "test_roc_auc": metrics.roc_auc,
            "test_f1_at_tuned_threshold": metrics.f1_at_threshold,
            "slice_rows_logged": float(len(slice_table)),
        }


def main() -> None:
    """CLI entrypoint for model evaluation."""
    result = evaluate_model()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
