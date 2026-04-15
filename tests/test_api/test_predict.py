from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_predict_invalid_input_type():
    """Verify that submitting strings for float features triggers a 422 Unprocessable Entity."""
    bad_payload = {
        "firm_id": "1234567890",
        "fiscal_year": 2026,
        "fiscal_period": "Q1",
        "total_assets": "INVALID_STRING",  # Expected float
        "total_liabilities": 500.0,
        "net_income": 100.0,
    }

    response = client.post("/predict", json=bad_payload, headers={"X-API-Key": "local-dev-key-123"})
    assert response.status_code == 422
    assert "detail" in response.json()
