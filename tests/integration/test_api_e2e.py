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


@pytest.fixture(autouse=True)
def mock_cloud_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocks all GCS, MLflow, and model loading to run tests completely offline."""

    # 1. Mock pandas read_parquet for company and alerts endpoints
    def mock_read_parquet(path: str, **kwargs: Any) -> pd.DataFrame:
        if "scores.parquet" in str(path):
            return pd.DataFrame(
                [
                    {
                        "firm_id": "0000123456",
                        "fiscal_year": 2025,
                        "fiscal_period": "Q4",
                        "company_name": "Test A",
                        "distress_probability": 0.8,
                        "confidence_interval_lower": 0.75,
                        "confidence_interval_upper": 0.85,
                        "active_signals": 2,
                    },
                    {
                        "firm_id": "0000123457",
                        "fiscal_year": 2025,
                        "fiscal_period": "Q4",
                        "company_name": "Test B",
                        "distress_probability": 0.9,
                        "confidence_interval_lower": 0.85,
                        "confidence_interval_upper": 0.95,
                        "active_signals": 3,
                    },
                    {
                        "firm_id": "0000123458",
                        "fiscal_year": 2025,
                        "fiscal_period": "Q4",
                        "company_name": "Test C",
                        "distress_probability": 0.1,
                        "confidence_interval_lower": 0.05,
                        "confidence_interval_upper": 0.15,
                        "active_signals": 0,
                    },
                ]
            )
        # Default fallback for explain helper
        return pd.DataFrame(
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

    monkeypatch.setattr("pandas.read_parquet", mock_read_parquet)
    monkeypatch.setattr("src.models.explain.pd.read_parquet", mock_read_parquet)

    # 2. Mock GCSFileSystem for JSON reads (health, drift)
    class MockFile:
        def __init__(self, content: str):
            self.content = content

        def __enter__(self) -> MockFile:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self) -> str:
            return self.content

    class MockGCS:
        def open(self, path: str, mode: str = "r") -> MockFile:
            if "manifest.json" in path:
                return MockFile(
                    '{"schema_version": "1.0", "model_name": "foresight_xgboost", "gcs_scores_path": "gs://..."}'
                )
            if "summary_latest.json" in path:
                return MockFile('{"dataset_drift": false, "report_url": "http://..."}')
            raise FileNotFoundError(f"Mock file not found: {path}")

    monkeypatch.setattr("gcsfs.GCSFileSystem", MockGCS)

    # 3. Mock the loaded ML models in memory to avoid lifespan load errors
    class MockModel:
        def predict(self, data: Any) -> list[float]:
            return [0.85]  # Returns high risk probability

    class MockScaler:
        def transform(self, data: Any) -> Any:
            return data

    import src.api.main

    src.api.main.ml_models["model"] = MockModel()
    src.api.main.ml_models["scaler"] = MockScaler()


# Define standard headers including the fallback API key from dependencies.py
HEADERS = {"X-API-Key": "local-dev-key-123"}


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info_endpoint(client: TestClient) -> None:
    response = client.get("/model/info", headers=HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["model_name"] == "foresight_xgboost"


def test_predict_endpoint(client: TestClient) -> None:
    response = client.post(
        "/predict",
        headers=HEADERS,
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
    assert "High" in payload["risk_level"]
    assert len(payload["top_features"]) == 3


def test_alerts_endpoint(client: TestClient) -> None:
    response = client.get("/alerts", params={"threshold": 0.75}, headers=HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload["threshold"] == 0.75
    assert len(payload["alerts"]) == 2  # Based on our mock data above


def test_company_endpoint(client: TestClient) -> None:
    response = client.get("/company/0000123456", headers=HEADERS)
    assert response.status_code == 200
    payload = response.json()
    # Verifies the new real endpoint returns a list of dictionaries directly
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["firm_id"] == "0000123456"


def test_drift_status_endpoint(client: TestClient) -> None:
    response = client.get("/drift/status", headers=HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_drift"] is False
    assert "report_url" in payload


def test_get_top_features_helper_with_gcs_mock() -> None:
    top_features = get_top_features("0000123456", "2025-Q4")
    assert len(top_features) == 3
    assert top_features[0]["feature"] == "net_income"
