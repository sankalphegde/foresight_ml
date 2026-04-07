"""Predict endpoint router."""
from fastapi import APIRouter
from src.api.schemas import PredictRequest, PredictResponse

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict_distress(request: PredictRequest):
    """Mocks a live prediction for Day 1 testing."""
    return {
        "distress_probability": 0.88,
        "risk_level": "High Risk",
        "top_features": [
            {"feature": "net_income", "shap_value": -0.45},
            {"feature": "total_assets", "shap_value": 0.22},
            {"feature": "total_liabilities", "shap_value": -0.15}
        ],
        "confidence_interval": [0.83, 0.93],
        "model_version": "v1",
        "scored_at": "2026-03-30T06:00:00+00:00"
    }