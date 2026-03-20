"""Model training and hyperparameter tuning pipeline."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from pandas.api.types import is_datetime64_any_dtype
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.config.settings import settings

DEFAULT_TRAIN_URI = "gs://financial-distress-data/splits/v1/train.parquet"
DEFAULT_VAL_URI = "gs://financial-distress-data/splits/v1/val.parquet"
DEFAULT_TEST_URI = "gs://financial-distress-data/splits/v1/test.parquet"
DEFAULT_SCALER_URI = "gs://financial-distress-data/splits/v1/scaler_pipeline.pkl"
DEFAULT_CLASS_WEIGHT_URI = "gs://financial-distress-data/splits/v1/scale_pos_weight.json"

DEFAULT_OUT_DIR = "artifacts/models"
DEFAULT_MODEL_URI = "gs://financial-distress-data/models/xgb_model.pkl"
DEFAULT_SCALER_OUT_URI = "gs://financial-distress-data/models/scaler_pipeline.pkl"
DEFAULT_TUNING_REPORT_URI = "gs://financial-distress-data/models/optuna_results.json"
DEFAULT_SENS_PLOT_URI = "gs://financial-distress-data/models/optuna_sensitivity.png"

LABEL_COL = os.getenv("LABEL_COL", "distress_label")


@dataclass
class SplitData:
    """Container for train/val/test dataframes."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _gcs_client() -> Any:
    """Create a Google Cloud Storage client."""
    from google.cloud import storage

    return storage.Client()


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse gs://bucket/path into (bucket, path)."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {gcs_uri}")
    stripped = gcs_uri.replace("gs://", "", 1)
    if "/" not in stripped:
        raise ValueError(f"Expected gs://bucket/path, got {gcs_uri}")
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


def _upload_local_to_gcs(local_path: Path, gcs_uri: str) -> None:
    """Upload a local file to a GCS object path."""
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))


def _load_split(gcs_uri: str) -> pd.DataFrame:
    """Load a parquet split from GCS."""
    return pd.read_parquet(gcs_uri)


def load_splits(train_uri: str, val_uri: str, test_uri: str) -> SplitData:
    """Load train/val/test splits from GCS."""
    return SplitData(
        train=_load_split(train_uri),
        val=_load_split(val_uri),
        test=_load_split(test_uri),
    )


def load_class_weight(gcs_uri: str | None, train_df: pd.DataFrame | None) -> float:
    """Load scale_pos_weight from GCS or compute from training labels."""
    if gcs_uri:
        try:
            tmp_path = Path("/tmp/scale_pos_weight.json")
            _download_gcs_to_local(gcs_uri, tmp_path)
            with open(tmp_path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "scale_pos_weight" in data:
                return float(data["scale_pos_weight"])
            if isinstance(data, int | float):
                return float(data)
        except Exception:
            pass

    if train_df is None:
        raise ValueError("scale_pos_weight not found and train_df not provided")
    _, labels = get_features_and_labels(train_df)
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    if pos == 0:
        return 1.0
    return max(1.0, neg / pos)


def load_scaler(gcs_uri: str) -> Path:
    """Download the fitted scaler pipeline from GCS."""
    local_path = Path("/tmp/scaler_pipeline.pkl")
    _download_gcs_to_local(gcs_uri, local_path)
    return local_path


def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a dataframe to all-numeric features."""
    out = df.copy()
    for col in out.columns:
        if is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype("int64")
    out = pd.get_dummies(out, dummy_na=True)
    return out


def get_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and labels."""
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")
    labels = df[LABEL_COL]
    features = df.drop(columns=[LABEL_COL])
    return features, labels


def prepare_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare aligned numeric feature matrices across splits."""
    x_train_raw, y_train = get_features_and_labels(train_df)
    x_val_raw, y_val = get_features_and_labels(val_df)
    x_test_raw, y_test = get_features_and_labels(test_df)

    x_train = _to_numeric_frame(x_train_raw)
    x_val = _to_numeric_frame(x_val_raw)
    x_test = _to_numeric_frame(x_test_raw)

    x_val = x_val.reindex(columns=x_train.columns, fill_value=0)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    return x_train, y_train, x_val, y_val, x_test, y_test


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scale_pos_weight: float,
) -> tuple[XGBClassifier, float]:
    """Train a baseline XGBoost model and return validation ROC-AUC."""
    x_train, y_train, x_val, y_val, _, _ = prepare_splits(train_df, val_df, val_df)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    preds = model.predict_proba(x_val)[:, 1]
    roc = roc_auc_score(y_val, preds)
    return model, float(roc)


def load_search_space(config_path: Path) -> dict[str, list[Any]]:
    """Load Optuna search space from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Invalid config format")
    return cfg


def _get_mlflow() -> Any:
    """Import mlflow lazily to avoid test-time dependency errors."""
    import mlflow

    return mlflow


def optuna_objective(
    trial: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scale_pos_weight: float,
    search_space: dict[str, list[Any]],
) -> float:
    """Optuna objective that maximizes validation ROC-AUC."""
    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            min(search_space["learning_rate"]),
            max(search_space["learning_rate"]),
            log=True,
        ),
        "max_depth": trial.suggest_categorical("max_depth", search_space["max_depth"]),
        "n_estimators": trial.suggest_categorical("n_estimators", search_space["n_estimators"]),
        "subsample": trial.suggest_categorical("subsample", search_space["subsample"]),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", search_space["colsample_bytree"]
        ),
        "min_child_weight": trial.suggest_categorical(
            "min_child_weight", search_space["min_child_weight"]
        ),
    }

    x_train, y_train, x_val, y_val, _, _ = prepare_splits(train_df, val_df, val_df)

    start = time.time()
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        **params,
    )
    model.fit(x_train, y_train)
    preds = model.predict_proba(x_val)[:, 1]
    roc = roc_auc_score(y_val, preds)
    duration = time.time() - start

    mlflow = _get_mlflow()
    mlflow.log_params(params)
    mlflow.log_metric("val_roc_auc", float(roc))
    mlflow.log_metric("train_time_sec", float(duration))

    return float(roc)


def run_optuna_tuning(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scale_pos_weight: float,
    search_space: dict[str, list[Any]],
    n_trials: int,
) -> Any:
    """Run Optuna tuning and return the study."""
    import optuna

    study = optuna.create_study(direction="maximize")
    for _ in range(n_trials):
        mlflow = _get_mlflow()
        with mlflow.start_run(nested=True):
            study.optimize(
                lambda t: optuna_objective(t, train_df, val_df, scale_pos_weight, search_space),
                n_trials=1,
                show_progress_bar=False,
            )
    return study


def save_sensitivity_plot(study: Any, out_path: Path) -> None:
    """Save ROC-AUC sensitivity plots for top hyperparameters."""
    import optuna

    importances = optuna.importance.get_param_importances(study)
    top_params = list(importances.keys())[:3]
    trials_df = study.trials_dataframe()

    if trials_df.empty:
        return

    fig, axes = plt.subplots(1, len(top_params), figsize=(5 * len(top_params), 4))
    if len(top_params) == 1:
        axes = [axes]

    for ax, param in zip(axes, top_params, strict=False):
        x = trials_df[f"params_{param}"]
        y = trials_df["value"]
        ax.scatter(x, y, alpha=0.7)
        ax.set_title(f"val ROC-AUC vs {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("val ROC-AUC")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Run baseline training, Optuna tuning, and artifact upload."""
    train_uri = os.getenv("TRAIN_URI", DEFAULT_TRAIN_URI)
    val_uri = os.getenv("VAL_URI", DEFAULT_VAL_URI)
    test_uri = os.getenv("TEST_URI", DEFAULT_TEST_URI)
    scaler_uri = os.getenv("SCALER_URI", DEFAULT_SCALER_URI)
    class_weight_uri = os.getenv("SCALE_POS_WEIGHT_URI", DEFAULT_CLASS_WEIGHT_URI)
    out_dir = Path(os.getenv("MODEL_OUT_DIR", DEFAULT_OUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    search_space = load_search_space(Path("configs/model/xgboost.yaml"))
    splits = load_splits(train_uri, val_uri, test_uri)
    scale_pos_weight = load_class_weight(class_weight_uri, splits.train)

    mlflow = _get_mlflow()
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    # Baseline
    with mlflow.start_run(run_name="baseline"):
        _, baseline_roc = train_baseline(splits.train, splits.val, scale_pos_weight)
        mlflow.log_metric("val_roc_auc", baseline_roc)

    # Optuna tuning
    n_trials = int(os.getenv("OPTUNA_TRIALS", "25"))
    with mlflow.start_run(run_name="optuna_tuning"):
        study = run_optuna_tuning(
            splits.train, splits.val, scale_pos_weight, search_space, n_trials
        )

    best_params = study.best_params

    # Retrain best model on full train split
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_splits(
        splits.train, splits.val, splits.test
    )

    full_train_x = pd.concat([x_train, x_val], axis=0)
    full_train_y = pd.concat([y_train, y_val], axis=0)

    best_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    best_model.fit(full_train_x, full_train_y)

    test_preds = best_model.predict_proba(x_test)[:, 1]
    test_roc = roc_auc_score(y_test, test_preds)

    report = {
        "baseline_val_roc": baseline_roc,
        "best_params": best_params,
        "test_roc_auc": float(test_roc),
    }

    report_path = out_dir / "optuna_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save model
    model_path = out_dir / "xgb_model.pkl"
    best_model.save_model(str(model_path))

    # Save sensitivity plot
    plot_path = out_dir / "optuna_sensitivity.png"
    save_sensitivity_plot(study, plot_path)

    # Upload artifacts to GCS
    _upload_local_to_gcs(report_path, os.getenv("MODEL_REPORT_URI", DEFAULT_TUNING_REPORT_URI))
    _upload_local_to_gcs(model_path, os.getenv("MODEL_ARTIFACT_URI", DEFAULT_MODEL_URI))
    _upload_local_to_gcs(plot_path, os.getenv("SENS_PLOT_URI", DEFAULT_SENS_PLOT_URI))

    # Upload scaler pipeline artifact
    scaler_local = load_scaler(scaler_uri)
    _upload_local_to_gcs(scaler_local, os.getenv("SCALER_OUT_URI", DEFAULT_SCALER_OUT_URI))


if __name__ == "__main__":
    main()
