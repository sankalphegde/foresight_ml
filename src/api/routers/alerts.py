"""Watchlist alerts endpoint router."""

import logging

import gcsfs
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import AlertItem, AlertsResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Alerts"])


@router.get("/alerts", response_model=AlertsResponse)
async def get_high_risk_alerts(
    threshold: float = Query(0.70, description="Minimum distress probability threshold"),
) -> AlertsResponse:
    """Returns a list of companies currently exceeding the risk threshold."""
    try:
        fs = gcsfs.GCSFileSystem()
        scores_path = "gs://financial-distress-data/inference/scores_v1.0/scores.parquet"

        df = pd.read_parquet(scores_path, filesystem=fs)

        # Filter for companies where the probability is above the requested threshold
        high_risk_df = df[df["distress_probability"] >= threshold]

        # Map the pandas rows into our Pydantic AlertItem schema
        alerts_list = []
        for _, row in high_risk_df.iterrows():
            alerts_list.append(
                AlertItem(
                    firm_id=str(row["firm_id"]),
                    company_name=str(row.get("company_name", "Unknown Company")),
                    distress_probability=float(row["distress_probability"]),
                    active_signals=int(row.get("active_signals", 0)),
                )
            )

        return AlertsResponse(threshold=threshold, alerts=alerts_list)

    except FileNotFoundError:
        logger.error("scores.parquet not found in GCS.")
        raise HTTPException(
            status_code=503, detail="Watchlist data is currently unavailable."
        ) from None
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while fetching alerts."
        ) from e
