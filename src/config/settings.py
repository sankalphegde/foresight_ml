"""Application configuration settings loaded from environment variables."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """Container for project-level configuration values."""
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    gcs_bucket: str = os.getenv("GCS_BUCKET", "")
    cleaned_path: str = "cleaned_data/final_v2/"
    panel_output_path: str = "features/panel_v1/panel.parquet"
    labeled_output_path: str = "features/labeled_v1/labeled_panel.parquet"
    prediction_horizon: int = int(os.getenv("PREDICTION_HORIZON", "1"))


settings = Settings()