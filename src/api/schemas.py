"""Pydantic schemas for the Foresight-ML API endpoints."""
from typing import List
from pydantic import BaseModel

from src.models.manifest_schema import ManifestSchema

class PredictRequest(BaseModel):
    """Request schema for POST /predict."""
# Request schema for POST /predict
class PredictRequest(BaseModel):
    firm_id: str
    fiscal_year: int
    fiscal_period: str
    total_assets: float
    total_liabilities: float
    net_income: float

class PredictResponse(BaseModel):
    """Response schema for POST /predict."""
# Response schema for POST /predict
class PredictResponse(BaseModel):
    distress_probability: float
    risk_level: str
    top_features: List[dict]
    confidence_interval: List[float]
    model_version: str
    scored_at: str

class AlertItem(BaseModel):
    """Schema for an individual alert in the watchlist."""
# Schema for an individual alert in the watchlist
class AlertItem(BaseModel):
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
# Response schema for GET /alerts
class AlertsResponse(BaseModel):
    threshold: float
    alerts: List[AlertItem]

# Response schema for GET /drift/status
class DriftStatusResponse(BaseModel):
    dataset_drift: bool
    drift_share: float
    report_url: str

class HealthResponse(BaseModel):
    """Response schema for GET /health."""
# Response schema for GET /health
class HealthResponse(BaseModel):
    status: str