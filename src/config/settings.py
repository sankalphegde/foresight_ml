"""Application configuration settings loaded from environment variables."""

import os
from dataclasses import dataclass


def _parse_year_range(value: str, default: tuple[int, int]) -> tuple[int, int]:
    """Parse year range from 'start,end' format."""
    if not value.strip():
        return default
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        return default
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        return default
    if start > end:
        return default
    return start, end


def _parse_year_list(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Parse year list from comma-separated values."""
    if not value.strip():
        return default
    parts = [part.strip() for part in value.split(",") if part.strip()]
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Container for project-level configuration values."""

    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    gcs_bucket: str = os.getenv("GCS_BUCKET", "")
    cleaned_path: str = "cleaned_data/final_v2/"
    panel_output_path: str = "features/panel_v1/panel.parquet"
    labeled_output_path: str = "features/labeled_v1/labeled_panel.parquet"
    prediction_horizon: int = int(os.getenv("PREDICTION_HORIZON", "1"))
    bigquery_features_table: str = os.getenv(
        "BIGQUERY_FEATURES_TABLE",
        "financial_distress_features.cleaned_engineered_features",
    )
    train_years: tuple[int, int] = _parse_year_range(
        os.getenv("TRAIN_YEARS", "2010,2019"),
        (2010, 2019),
    )
    val_years: tuple[int, int] = _parse_year_range(
        os.getenv("VAL_YEARS", "2020,2021"),
        (2020, 2021),
    )
    test_years: tuple[int, int] = _parse_year_range(
        os.getenv("TEST_YEARS", "2022,2023"),
        (2022, 2023),
    )
    exclude_years: tuple[int, ...] = _parse_year_list(
        os.getenv("EXCLUDE_YEARS", "2009"),
        (2009,),
    )
    local_splits_dir: str = os.getenv("LOCAL_SPLITS_DIR", "data/splits")
    splits_output_path: str = os.getenv("SPLITS_OUTPUT_PATH", "splits/v1/")
    scaler_output_path: str = os.getenv("SCALER_OUTPUT_PATH", "splits/v1/scaler_pipeline.pkl")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "foresight-training")
    mlflow_model_name: str = os.getenv("MLFLOW_MODEL_NAME", "foresight-distress-model")
    mlflow_model_alias: str = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


settings = Settings()
