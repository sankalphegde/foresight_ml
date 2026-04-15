"""Data drift monitoring using Evidently AI.

Compares current labeled panel against training reference data.
Generates HTML report + JSON summary and writes a retrain flag
to GCS if drift is detected.

GCS outputs:
    monitoring/drift_reports/report_{date}.html
    monitoring/drift_reports/summary_{date}.json
    monitoring/drift_reports/summary_latest.json
    monitoring/triggers/retrain_flag.json  (only if drift detected)
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from src.utils.logging import get_logger

logger = get_logger(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "financial-distress-data")
REFERENCE_PATH = f"gs://{GCS_BUCKET}/splits/v1/train.parquet"
CURRENT_PATH = f"gs://{GCS_BUCKET}/features/labeled_v1/labeled_panel.parquet"
DRIFT_REPORTS_PREFIX = "monitoring/drift_reports"
RETRAIN_FLAG_PATH = "monitoring/triggers/retrain_flag.json"
PSI_THRESHOLD = 0.25

# Columns to exclude from drift analysis
EXCLUDE_COLS = [
    "firm_id",
    "date",
    "fiscal_year",
    "fiscal_period",
    "quarter_key",
    "filed_date",
    "distress_label",
]


def _upload_to_gcs(local_path: Path, gcs_path: str) -> None:
    """Upload a local file to GCS."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    bucket.blob(gcs_path).upload_from_filename(str(local_path))
    logger.info("Uploaded %s to gs://%s/%s", local_path.name, GCS_BUCKET, gcs_path)


def _write_retrain_flag(reason: str, drifted_features: list[str]) -> None:
    """Write retrain flag JSON to GCS."""
    from google.cloud import storage

    flag = {
        "triggered_at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "drifted_features": drifted_features,
    }
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    bucket.blob(RETRAIN_FLAG_PATH).upload_from_string(
        json.dumps(flag, indent=2),
        content_type="application/json",
    )
    logger.warning(
        "Retrain flag written to gs://%s/%s — reason: %s",
        GCS_BUCKET,
        RETRAIN_FLAG_PATH,
        reason,
    )


def run_drift_monitor() -> dict:
    """Run Evidently drift analysis and write reports to GCS.

    Returns:
        Summary dict with drift status and metrics.
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    logger.info("Loading reference dataset from %s", REFERENCE_PATH)
    reference_df = pd.read_parquet(REFERENCE_PATH)

    logger.info("Loading current dataset from %s", CURRENT_PATH)
    current_df = pd.read_parquet(CURRENT_PATH)

    # Select only numeric feature columns present in both datasets
    feature_cols = [
        c
        for c in reference_df.columns
        if c not in EXCLUDE_COLS
        and pd.api.types.is_numeric_dtype(reference_df[c])
        and c in current_df.columns
    ]
    logger.info("Running drift analysis on %d features", len(feature_cols))

    ref = reference_df[feature_cols].copy()
    cur = current_df[feature_cols].copy()

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])

    # Save HTML report locally then upload
    tmp_dir = Path("/tmp/drift_reports")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    report.run(reference_data=ref, current_data=cur)

    html_path = tmp_dir / f"report_{today}.html"
    report.save_html(str(html_path))
    _upload_to_gcs(html_path, f"{DRIFT_REPORTS_PREFIX}/report_{today}.html")

    report_dict = report.as_dict()
    metrics = report_dict.get("metrics", [])

    count_metric: dict[str, Any] = next(
        (
            m
            for m in metrics
            if str(m.get("metric_name", "")).startswith("DriftedColumnsCount")
            or str(m.get("config", {}).get("type", "")).endswith("DriftedColumnsCount")
        ),
        {},
    )
    count_value = count_metric.get("value", {})
    drift_share = float(count_value.get("share", 0.0) or 0.0)
    drift_threshold = float(count_metric.get("config", {}).get("drift_share", 0.5) or 0.5)
    dataset_drift = drift_share >= drift_threshold

    drifted_features = []
    for metric in metrics:
        if not str(metric.get("metric_name", "")).startswith("ValueDrift"):
            continue
        column = str(metric.get("config", {}).get("column", ""))
        threshold = float(metric.get("config", {}).get("threshold", 0.05) or 0.05)
        raw_value = metric.get("value")
        try:
            drift_score = float(raw_value)
        except (TypeError, ValueError):
            continue

        if column and drift_score <= threshold:
            drifted_features.append(column)

    summary = {
        "date": today,
        "dataset_drift": dataset_drift,
        "drift_share": round(drift_share, 4),
        "n_drifted_features": len(drifted_features),
        "drifted_features": drifted_features[:10],  # top 10
        "n_features_analyzed": len(feature_cols),
        "retrain_triggered": False,
    }

    # Write JSON summaries
    summary_path = tmp_dir / f"summary_{today}.json"
    latest_path = tmp_dir / "summary_latest.json"

    for path in [summary_path, latest_path]:
        path.write_text(json.dumps(summary, indent=2))

    _upload_to_gcs(summary_path, f"{DRIFT_REPORTS_PREFIX}/summary_{today}.json")
    _upload_to_gcs(latest_path, f"{DRIFT_REPORTS_PREFIX}/summary_latest.json")

    # Trigger retraining if drift detected
    if dataset_drift:
        logger.warning(
            "Dataset drift detected — %d/%d features drifted (%.1f%%)",
            len(drifted_features),
            len(feature_cols),
            drift_share * 100,
        )
        _write_retrain_flag(
            reason="dataset_drift",
            drifted_features=drifted_features,
        )
        summary["retrain_triggered"] = True
    else:
        logger.info(
            "No dataset drift detected — %d/%d features drifted (%.1f%%)",
            len(drifted_features),
            len(feature_cols),
            drift_share * 100,
        )

    logger.info("Drift monitoring complete — summary_latest.json written to GCS")
    return summary
