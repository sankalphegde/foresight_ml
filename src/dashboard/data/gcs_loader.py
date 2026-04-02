"""GCS data loader for the Streamlit dashboard.

Loads scores, SHAP values, labeled panel, manifest, drift reports,
and model metadata from GCS. All loaders are cached with a 5-minute TTL
so the dashboard stays responsive.

Gracefully returns empty DataFrames / default dicts when files don't exist yet
(e.g., scores parquet before Person 5 runs batch inference).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GCS paths — configurable via st.secrets or environment
# ---------------------------------------------------------------------------
GCS_BUCKET = "financial-distress-data"

SCORES_URI = f"gs://{GCS_BUCKET}/inference/scores_v1/scores.parquet"
SHAP_URI = f"gs://{GCS_BUCKET}/shap/shap_values.parquet"
LABELED_PANEL_URI = f"gs://{GCS_BUCKET}/features/labeled_v1/labeled_panel.parquet"
MANIFEST_URI = f"gs://{GCS_BUCKET}/inference/scores_v1/manifest.json"
OPTUNA_URI = f"gs://{GCS_BUCKET}/models/optuna_results.json"
DRIFT_SUMMARY_URI = f"gs://{GCS_BUCKET}/monitoring/drift_reports/summary_latest.json"
SLICE_PERF_URI = f"gs://{GCS_BUCKET}/mlflow/artifacts/slice_metrics/slice_performance.csv"


# ---------------------------------------------------------------------------
# Helper: safe GCS JSON read
# ---------------------------------------------------------------------------


def _read_gcs_json(uri: str) -> dict | None:
    """Read a JSON file from GCS, return None on failure."""
    try:
        from google.cloud import storage

        bucket_name = uri.replace("gs://", "").split("/")[0]
        blob_path = "/".join(uri.replace("gs://", "").split("/")[1:])
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        content = blob.download_as_text()
        return json.loads(content)
    except Exception as e:
        log.warning("Could not read %s: %s", uri, e)
        return None


# ---------------------------------------------------------------------------
# Loaders — each cached independently
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner="Loading scores...")
def load_scores() -> pd.DataFrame:
    """Load batch inference scores from GCS.

    Returns empty DataFrame if scores don't exist yet.
    """
    try:
        df = pd.read_parquet(SCORES_URI)
        log.info("Loaded scores: %d rows", len(df))
        return df
    except Exception as e:
        log.warning("Scores not available: %s", e)
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="Loading SHAP values...")
def load_shap_values() -> pd.DataFrame:
    """Load precomputed SHAP values parquet from GCS."""
    try:
        df = pd.read_parquet(SHAP_URI)
        log.info("Loaded SHAP values: %d rows", len(df))
        return df
    except Exception as e:
        log.warning("SHAP values not available: %s", e)
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="Loading company data...")
def load_labeled_panel() -> pd.DataFrame:
    """Load the labeled panel (all company-quarter rows with features + labels)."""
    try:
        df = pd.read_parquet(LABELED_PANEL_URI)
        log.info("Loaded labeled panel: %d rows", len(df))
        return df
    except Exception as e:
        log.warning("Labeled panel not available: %s", e)
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="Loading model info...")
def load_manifest() -> dict:
    """Load manifest.json from GCS.

    Returns default dict if manifest doesn't exist yet.
    """
    data = _read_gcs_json(MANIFEST_URI)
    if data:
        return data
    return {
        "schema_version": "1.0",
        "model_name": "foresight-xgboost",
        "model_version": "v1",
        "mlflow_run_id": "pending",
        "trained_at": "2026-03-15T10:00:00+00:00",
        "scored_at": "pending",
        "roc_auc": 0.9769,
        "prediction_horizon": 1,
        "features_used": [],
        "row_count": 0,
        "gcs_scores_path": SCORES_URI,
        "inference_duration_seconds": 0,
    }


@st.cache_data(ttl=300)
def load_optuna_results() -> dict:
    """Load Optuna training results from GCS."""
    data = _read_gcs_json(OPTUNA_URI)
    if data:
        return data
    return {
        "baseline_val_roc": 0.0,
        "best_params": {},
        "test_roc_auc": 0.0,
    }


@st.cache_data(ttl=300)
def load_drift_summary() -> dict:
    """Load latest drift monitoring summary from GCS.

    Returns default (no drift) if report doesn't exist yet.
    """
    data = _read_gcs_json(DRIFT_SUMMARY_URI)
    if data:
        return data
    return {
        "drift_detected": False,
        "report_date": None,
        "drifted_features": [],
        "total_features_checked": 0,
    }


@st.cache_data(ttl=300)
def load_slice_performance() -> pd.DataFrame:
    """Load per-slice performance table from GCS."""
    try:
        df = pd.read_parquet(SLICE_PERF_URI)
        return df
    except Exception:
        try:
            df = pd.read_csv(SLICE_PERF_URI.replace(".parquet", ".csv"))
            return df
        except Exception as e:
            log.warning("Slice performance not available: %s", e)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Derived data helpers
# ---------------------------------------------------------------------------


def get_company_list(panel: pd.DataFrame) -> pd.DataFrame:
    """Get unique company list with latest data for search/filter.

    Returns DataFrame with firm_id, latest fiscal_year, fiscal_period,
    and key financial metrics from the most recent quarter.
    """
    if panel.empty:
        return pd.DataFrame()

    panel = panel.sort_values(["firm_id", "fiscal_year", "fiscal_period"])
    latest = panel.groupby("firm_id").last().reset_index()

    cols = ["firm_id", "fiscal_year", "fiscal_period"]
    optional = [
        "total_assets", "net_income", "distress_label",
        "company_size_bucket", "sector_proxy",
    ]
    for c in optional:
        if c in latest.columns:
            cols.append(c)

    return latest[cols]


def get_company_history(
    panel: pd.DataFrame, firm_id: str
) -> pd.DataFrame:
    """Get all quarterly rows for a specific company, sorted by time."""
    if panel.empty:
        return pd.DataFrame()
    mask = panel["firm_id"] == firm_id
    return panel[mask].sort_values(["fiscal_year", "fiscal_period"]).copy()


def get_shap_for_company(
    shap_df: pd.DataFrame, firm_id: str, fiscal_year: int | None = None
) -> pd.DataFrame:
    """Get SHAP rows for a specific company, optionally filtered by year."""
    if shap_df.empty or "firm_id" not in shap_df.columns:
        return pd.DataFrame()
    mask = shap_df["firm_id"] == firm_id
    if fiscal_year is not None and "fiscal_year" in shap_df.columns:
        mask = mask & (shap_df["fiscal_year"] == fiscal_year)
    return shap_df[mask].copy()