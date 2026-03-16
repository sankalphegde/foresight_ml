"""Application configuration settings loaded from environment variables."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Container for project-level configuration values."""

    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    gcs_bucket: str = os.getenv("GCS_BUCKET", "")
    cleaned_path: str = "cleaned_data/final_v2/"
    panel_output_path: str = "features/panel_v1/panel.parquet"
    labeled_output_path: str = "features/labeled_v1/labeled_panel.parquet"
    prediction_horizon: int = int(os.getenv("PREDICTION_HORIZON", "1"))
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "foresight-training")
    mlflow_model_name: str = os.getenv("MLFLOW_MODEL_NAME", "foresight-distress-model")
    mlflow_model_alias: str = os.getenv("MLFLOW_MODEL_ALIAS", "champion")

    # --- Data splitting config ---
    bigquery_features_table: str = os.getenv(
        "BQ_FEATURES_TABLE",
        "foresight_ml.cleaned_engineered_features",
    )
    local_splits_dir: str = os.getenv("LOCAL_SPLITS_DIR", "data/splits")
    splits_output_path: str = "splits/v1/"
    scaler_output_path: str = "splits/v1/scaler_pipeline.pkl"
    split_report_path: str = "splits/v1/split_report.json"
    class_weights_path: str = "splits/v1/class_weights.json"

    # Time-based split boundaries
    train_years: tuple[int, int] = (2010, 2019)
    val_years: tuple[int, int] = (2020, 2021)
    test_years: tuple[int, int] = (2022, 2023)
    exclude_years: tuple[int, ...] = (2009,)


settings = Settings()
