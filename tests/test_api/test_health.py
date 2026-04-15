from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_ping():
    """Verify the unprotected health check returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("gcsfs.GCSFileSystem.open")
def test_model_info_503_on_gcs_failure(mock_open):
    """Verify that a missing manifest triggers a 503 Service Unavailable."""
    mock_open.side_effect = FileNotFoundError
    response = client.get("/model/info", headers={"X-API-Key": "local-dev-key-123"})
    assert response.status_code == 503
    assert "not found" in response.json()["detail"].lower()
