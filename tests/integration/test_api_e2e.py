from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.explain import get_top_features


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def mock_external_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock GCS parquet read path and MLflow loader used by explain helpers."""

    frame = pd.DataFrame(
        [
            {
                "firm_id": "0000123456",
                "fiscal_year": 2025,
                "fiscal_period": "Q4",
                "top_features_json": json.dumps(
                    [
                        {"feature": "net_income", "shap_value": -0.45, "rank": 1},
                        {"feature": "total_assets", "shap_value": 0.22, "rank": 2},
                        {"feature": "total_liabilities", "shap_value": -0.15, "rank": 3},
                    ]
                ),
            }
        ]
    )
    monkeypatch.setattr("src.models.explain.pd.read_parquet", lambda *_args, **_kwargs: frame)

    class DummyMlflow:
        def __getattr__(self, _name: str) -> Any:
            def _noop(*_args: Any, **_kwargs: Any) -> None:
                return None

            return _noop

    monkeypatch.setattr("src.models.explain._get_mlflow", lambda: DummyMlflow())


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info_endpoint(client: TestClient) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["model_name"] == "foresight_xgboost"
    assert payload["gcs_scores_path"].startswith("gs://")


def test_predict_endpoint(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "firm_id": "0000123456",
            "fiscal_year": 2025,
            "fiscal_period": "Q4",
            "total_assets": 100.0,
            "total_liabilities": 50.0,
            "net_income": 10.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_level"] == "High Risk"
    assert len(payload["top_features"]) == 3


def test_alerts_endpoint(client: TestClient) -> None:
    response = client.get("/alerts", params={"threshold": 0.75})
    assert response.status_code == 200
    payload = response.json()
    assert payload["threshold"] == 0.75
    assert len(payload["alerts"]) == 2


def test_company_endpoint(client: TestClient) -> None:
    response = client.get("/company/0000123456")
    assert response.status_code == 200
    payload = response.json()
    assert payload["cik"] == "0000123456"
    assert len(payload["history"]) == 4


def test_drift_status_endpoint(client: TestClient) -> None:
    response = client.get("/drift/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_drift"] is False
    assert "report_url" in payload


def test_get_top_features_helper_with_gcs_mock(mock_external_services: None) -> None:
    top_features = get_top_features("0000123456", "2025-Q4")
    assert len(top_features) == 3
    assert top_features[0]["feature"] == "net_income"
