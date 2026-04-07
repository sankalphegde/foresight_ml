"""System health endpoint router."""
from fastapi import APIRouter
from src.api.schemas import HealthResponse
from src.models.manifest_schema import ManifestSchema

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check() -> dict:
    """Simple health check endpoint for Cloud Run."""
    return {"status": "healthy"}

@router.get("/model/info", response_model=ManifestSchema)
def get_model_info() -> dict:
    """Returns the mock manifest.json schema matching Palak's structure."""
    return {
        "schema_version": "1.0",
        "model_name": "foresight_xgboost",
        "model_version": "v1",
        "mlflow_run_id": "abc123def456",
        "trained_at": "2026-03-15T10:00:00+00:00",
        "scored_at": "2026-03-30T06:00:00+00:00",
        "roc_auc": 0.9769,
        "prediction_horizon": 1,
        "features_used": ["total_assets", "net_income"],
        "row_count": 48291,
        "gcs_scores_path": "gs://bucket/inference/v1/scores.parquet",
        "inference_duration_seconds": 42.7,
    }