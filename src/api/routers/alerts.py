"""Watchlist alerts endpoint router."""
from fastapi import APIRouter
from src.api.schemas import AlertsResponse

router = APIRouter()

@router.get("/alerts", response_model=AlertsResponse)
def get_high_risk_alerts(threshold: float = 0.7) -> dict:
    """Mocks high-risk watchlist data for Nandana's dashboard."""
    return {
        "threshold": threshold,
        "alerts": [
            {
                "firm_id": "0000001750",
                "company_name": "AAR CORP",
                "distress_probability": 0.88,
                "active_signals": 3
            },
            {
                "firm_id": "0000001800",
                "company_name": "ABBOTT LABS",
                "distress_probability": 0.75,
                "active_signals": 2
            }
        ]
    }