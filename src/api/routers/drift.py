"""Model drift status endpoint router."""

import json
import logging
from typing import Any, cast

import gcsfs
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Drift"])


@router.get("/drift/status")
async def get_drift_status() -> dict[str, Any]:
    """Reads the latest drift summary report from GCS."""
    try:
        fs = gcsfs.GCSFileSystem()
        report_path = "gs://financial-distress-data/monitoring/drift_reports/summary_latest.json"

        with fs.open(report_path, "r") as f:
            drift_data = json.load(f)

        return cast(dict[str, Any], drift_data)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Drift report not found.") from None
    except Exception as e:
        logger.error(f"Error reading drift report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
