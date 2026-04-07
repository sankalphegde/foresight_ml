"""Model drift status endpoint router."""
from fastapi import APIRouter
from src.api.schemas import DriftStatusResponse

router = APIRouter()

@router.get("/drift/status", response_model=DriftStatusResponse)
def get_drift_status():
    """Mocks the data drift status from Evidently AI."""
    return {
        "dataset_drift": False,
        "drift_share": 0.15,
        "report_url": "gs://financial-distress-data/monitoring/drift_reports/summary_latest.html"
    }