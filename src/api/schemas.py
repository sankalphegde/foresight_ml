"""Pydantic schemas for the Foresight-ML API endpoints."""
from typing import List
from pydantic import BaseModel

# Importing Palak's schema from her files
from src.models.manifest_schema import ManifestSchema

class PredictRequest(BaseModel):
    """Request schema for POST /predict."""
    firm_id: str
    fiscal_year: int
    fiscal_period: str
    total_assets: float
    total_liabilities: float
    net_income: float

class PredictResponse(BaseModel):
    """Response schema for POST /predict."""
    distress_probability: float
    risk_level: str
    top_features: List[dict]
    confidence_interval: List[float]
    model_version: str
    scored_at: str

class AlertItem(BaseModel):
    """Schema for an individual alert in the watchlist."""
    firm_id: str
    company_name: str
    distress_probability: float
    active_signals: int

class AlertsResponse(BaseModel):
    """Response schema for GET /alerts."""
    threshold: float
    alerts: List[AlertItem]

class DriftStatusResponse(BaseModel):
    """Response schema for GET /drift/status."""
    dataset_drift: bool
    drift_share: float
    report_url: str

class HealthResponse(BaseModel):
    """Response schema for GET /health."""
    status: str