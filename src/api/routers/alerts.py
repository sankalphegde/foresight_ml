"""Watchlist alerts endpoint router."""

import logging
from pathlib import Path

import gcsfs
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import AlertItem, AlertsResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Alerts"])

_LOCAL_REF = (
    Path(__file__).parent.parent.parent.parent / "artifacts" / "reference" / "company_names.csv"
)
_GCS_REF = "gs://financial-distress-data/reference/company_names.csv"


def _load_name_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Load CIK → name and CIK → ticker maps. Tries local first, then GCS."""
    try:
        if _LOCAL_REF.exists():
            ref = pd.read_csv(_LOCAL_REF, dtype=str).fillna("")
        else:
            ref = (
                pd.read_parquet(_GCS_REF)
                if _GCS_REF.endswith(".parquet")
                else pd.read_csv(_GCS_REF, dtype=str, storage_options={"token": "google_default"})
            )
            ref = ref.fillna("")
        return dict(zip(ref["firm_id"], ref["name"], strict=False)), dict(
            zip(ref["firm_id"], ref["ticker"], strict=False)
        )
    except Exception as e:
        logger.warning("Could not load company name reference: %s", e)
        return {}, {}


@router.get("/alerts", response_model=AlertsResponse)
async def get_high_risk_alerts(
    threshold: float = Query(0.70, description="Minimum distress probability threshold"),
) -> AlertsResponse:
    """Returns a list of companies currently exceeding the risk threshold."""
    try:
        fs = gcsfs.GCSFileSystem()
        scores_path = "gs://financial-distress-data/inference/scores_v1.0/scores.parquet"

        df = pd.read_parquet(scores_path, filesystem=fs)

        # Deduplicate: keep the highest distress probability per firm
        df = df.sort_values("distress_probability", ascending=False).drop_duplicates(
            subset=["firm_id"], keep="first"
        )

        # Load company name + ticker maps
        name_map, ticker_map = _load_name_maps()

        # Filter above threshold
        high_risk_df = df[df["distress_probability"] >= threshold]

        alerts_list = []
        for _, row in high_risk_df.iterrows():
            firm_id = str(row["firm_id"])
            name = name_map.get(firm_id, "").strip()
            ticker = ticker_map.get(firm_id, "").strip()
            if not name:
                name = f"CIK {firm_id}"
            display_name = f"{name} ({ticker})" if ticker else name
            alerts_list.append(
                AlertItem(
                    firm_id=firm_id,
                    company_name=display_name,
                    distress_probability=float(row["distress_probability"]),
                    active_signals=int(row.get("active_signals", 0)),
                )
            )

        alerts_list.sort(key=lambda x: x.distress_probability, reverse=True)

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
