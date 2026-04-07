"""Company data endpoint router."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/company/{cik}")
def get_company_history(cik: str):
    """Mocks historical distress scores for a specific company."""
    return {
        "cik": cik,
        "history": [
            {"quarter": "2025-Q1", "probability": 0.12},
            {"quarter": "2025-Q2", "probability": 0.15},
            {"quarter": "2025-Q3", "probability": 0.45},
            {"quarter": "2025-Q4", "probability": 0.88},
        ]
    }