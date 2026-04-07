"""API client for the Foresight-ML FastAPI service.

Wraps all API calls with error handling. Falls back gracefully
when the API is not yet deployed (returns None so the dashboard
can use GCS data instead).
"""

from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)

API_BASE_URL = os.getenv(
    "FORESIGHT_API_URL",
    "https://foresight-api-6ool3rlbea-uc.a.run.app",
)
API_TIMEOUT = 10  # seconds


def _get(endpoint: str, params: dict | None = None) -> dict | None:  # type: ignore[type-arg]
    """Make a GET request to the API. Returns None on failure."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        resp = requests.get(url, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("API call failed [GET %s]: %s", endpoint, e)
        return None


def _post(endpoint: str, payload: dict) -> dict | None:  # type: ignore[type-arg]
    """Make a POST request to the API. Returns None on failure."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        resp = requests.post(url, json=payload, timeout=API_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("API call failed [POST %s]: %s", endpoint, e)
        return None


# ---------------------------------------------------------------------------
# Public API methods
# ---------------------------------------------------------------------------


def check_health() -> dict | None:
    """GET /health — check if API is running."""
    return _get("/health")


def get_model_info() -> dict | None:
    """GET /model/info — get current production model metadata."""
    return _get("/model/info")


def predict(cik: str) -> dict | None:
    """POST /predict — get distress prediction for a company."""
    return _post("/predict", {"cik": cik})


def get_company(cik: str) -> dict | None:
    """GET /company/{cik} — get historical scores for a company."""
    return _get(f"/company/{cik}")


def get_alerts(threshold: float = 0.70) -> dict | None:
    """GET /alerts — get all companies above risk threshold."""
    return _get("/alerts", params={"threshold": threshold})


def get_drift_status() -> dict | None:
    """GET /drift/status — get latest drift monitoring status."""
    return _get("/drift/status")


def is_api_available() -> bool:
    """Quick check if the API is reachable."""
    result = check_health()
    return result is not None
