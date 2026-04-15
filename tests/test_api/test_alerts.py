from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@patch("pandas.read_parquet")
@patch("gcsfs.GCSFileSystem")
def test_alerts_threshold_filtering(mock_gcs, mock_read):
    """Verify that only companies exceeding the distress threshold are returned."""
    # Mock a dataframe with mixed risk levels
    mock_read.return_value = pd.DataFrame(
        [
            {"firm_id": "0001", "distress_probability": 0.85},
            {"firm_id": "0002", "distress_probability": 0.40},
            {"firm_id": "0003", "distress_probability": 0.72},
        ]
    )

    response = client.get(
        "/alerts", params={"threshold": 0.70}, headers={"X-API-Key": "local-dev-key-123"}
    )

    assert response.status_code == 200
    alerts = response.json()["alerts"]
    assert len(alerts) == 2  # Should only return firm 0001 and 0003
