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


settings = Settings()
