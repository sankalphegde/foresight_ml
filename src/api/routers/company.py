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
        scores_path = "gs://financial-distress-data/inference/scores_v1.0/scores.parquet"

        df = pd.read_parquet(scores_path, filesystem=fs)

        company_data = df[df["firm_id"] == cik].copy()

        if company_data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for CIK: {cik}")

        # Deduplicate by firm + quarter, keep highest distress_probability
        company_data = (
            company_data
            .sort_values("distress_probability", ascending=False)
            .drop_duplicates(subset=["firm_id", "fiscal_year", "fiscal_period"], keep="first")
            .sort_values(["fiscal_year", "fiscal_period"])
        )

        # Convert timestamps to strings for JSON serialization
        for col in company_data.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
            company_data[col] = company_data[col].astype(str)

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
