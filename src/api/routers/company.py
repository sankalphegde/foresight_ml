"""Company data endpoint router."""

import logging
from typing import Any, cast

import gcsfs
import pandas as pd
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Company Info"])


@router.get("/company/{cik}")
async def get_company_history(cik: str) -> list[dict[str, Any]]:
    """Fetches historical distress scores for a specific company (CIK)."""
    try:
        fs = gcsfs.GCSFileSystem()
        # This path relies on Palak's batch inference output
        scores_path = "gs://financial-distress-data/inference/scores_v1.0/scores.parquet"

        # Read the parquet file directly from GCS into a Pandas DataFrame
        df = pd.read_parquet(scores_path, filesystem=fs)

        company_data = df[df["firm_id"] == cik]

        if company_data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for CIK: {cik}")

        # Convert the filtered DataFrame to a list of dictionaries for JSON output
        return cast(list[dict[str, Any]], company_data.to_dict(orient="records"))

    except FileNotFoundError:
        logger.error("scores.parquet not found in GCS. Batch inference may not have run.")
        raise HTTPException(
            status_code=503, detail="Historical scores are currently unavailable."
        ) from None
    except Exception as e:
        logger.error(f"Error fetching company data: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while fetching company data."
        ) from e
