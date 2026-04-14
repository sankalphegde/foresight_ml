"""GCS data loader for the Streamlit dashboard.

Loads predictions, SHAP values, labeled panel, manifest, drift reports,
company names, and model metadata. All loaders are cached with TTL
so the dashboard stays responsive.

Gracefully returns empty DataFrames / default dicts when data is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GCS paths — single source of truth
# ---------------------------------------------------------------------------
GCS_BUCKET = os.getenv("GCS_BUCKET", "financial-distress-data")

SCORES_URI = f"gs://{GCS_BUCKET}/inference/scores_v1.0/scores.parquet"
SHAP_URI = f"gs://{GCS_BUCKET}/shap/shap_values.parquet"
LABELED_PANEL_URI = f"gs://{GCS_BUCKET}/features/labeled_v1/labeled_panel.parquet"
MANIFEST_URI = f"gs://{GCS_BUCKET}/inference/scores_v1.0/manifest.json"
OPTUNA_URI = f"gs://{GCS_BUCKET}/models/optuna_results.json"
DRIFT_SUMMARY_URI = f"gs://{GCS_BUCKET}/monitoring/drift_reports/summary_latest.json"
SLICE_PERF_URI = f"gs://{GCS_BUCKET}/mlflow/artifacts/slice_metrics/slice_performance.csv"

# Local fallback paths
LOCAL_MODEL = Path("artifacts/models/xgb_model.pkl")
LOCAL_TEST = Path("artifacts/splits/test.parquet")
LOCAL_COMPANY_NAMES = Path("artifacts/reference/company_names.csv")
LOCAL_COMPANY_REF = Path("artifacts/reference/companies.csv")

# Default manifest when GCS is unavailable
DEFAULT_MANIFEST = {
    "schema_version": "1.0",
    "model_name": "foresight_xgboost",
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_gcs_json(uri: str) -> dict | None:
    """Read a JSON file from GCS. Returns None on any failure."""
    try:
        from google.cloud import storage

        bucket_name = uri.replace("gs://", "").split("/")[0]
        blob_path = "/".join(uri.replace("gs://", "").split("/")[1:])
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
        return json.loads(blob.download_as_text())  # type: ignore[no-any-return]
    except Exception as e:
        log.warning("Could not read %s: %s", uri, e)
        return None


def _safe_read_parquet(
    uri: str,
    label: str = "data",
    columns: list[str] | None = None,
    filters: list[tuple[str, str, object]] | None = None,
) -> pd.DataFrame:
    """Read a parquet file, returning empty DataFrame on failure."""
    try:
        df = pd.read_parquet(uri, columns=columns, filters=filters)
        log.info("Loaded %s: %d rows", label, len(df))
        return df
    except Exception as e:
        log.warning("%s not available: %s", label, e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Primary data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner="Loading scores...")
def load_scores() -> pd.DataFrame:
    """Load batch inference scores from GCS."""
    return _safe_read_parquet(SCORES_URI, "scores")


@st.cache_data(ttl=300, show_spinner="Loading SHAP values...")
def load_shap_values() -> pd.DataFrame:
    """Load precomputed SHAP values parquet from GCS."""
    return _safe_read_parquet(SHAP_URI, "SHAP values")


@st.cache_data(ttl=300, show_spinner="Loading SHAP values...")
def load_shap_for_company(firm_id: str) -> pd.DataFrame:
    """Load SHAP rows for a single company to reduce memory pressure."""
    return _safe_read_parquet(
        SHAP_URI,
        f"SHAP values for {firm_id}",
        filters=[("firm_id", "==", firm_id)],
    )


@st.cache_data(ttl=300, show_spinner="Loading company data...")
def load_labeled_panel() -> pd.DataFrame:
    """Load the full labeled panel (all company-quarter rows)."""
    return _safe_read_parquet(LABELED_PANEL_URI, "labeled panel")


@st.cache_data(ttl=300, show_spinner="Loading company data...")
def load_company_history_rows(firm_id: str) -> pd.DataFrame:
    """Load labeled panel rows for a single company to reduce memory pressure."""
    return _safe_read_parquet(
        LABELED_PANEL_URI,
        f"labeled panel for {firm_id}",
        filters=[("firm_id", "==", firm_id)],
    )


@st.cache_data(ttl=300)
def load_panel_firm_ids() -> list[str]:
    """Load distinct firm IDs from panel using only the firm_id column."""
    panel_ids = _safe_read_parquet(
        LABELED_PANEL_URI,
        "panel firm ids",
        columns=["firm_id"],
    )
    if panel_ids.empty or "firm_id" not in panel_ids.columns:
        return []
    return sorted(panel_ids["firm_id"].dropna().astype(str).unique().tolist())


@st.cache_data(ttl=300, show_spinner="Loading model info...")
def load_manifest() -> dict:
    """Load manifest.json from GCS, with sensible defaults."""
    return _read_gcs_json(MANIFEST_URI) or DEFAULT_MANIFEST


@st.cache_data(ttl=300)
def load_optuna_results() -> dict:
    """Load Optuna training results from GCS."""
    return _read_gcs_json(OPTUNA_URI) or {
        "baseline_val_roc": 0.0,
        "best_params": {},
        "test_roc_auc": 0.0,
    }


@st.cache_data(ttl=300)
def load_drift_summary() -> dict:
    """Load latest drift monitoring summary from GCS."""
    return _read_gcs_json(DRIFT_SUMMARY_URI) or {
        "drift_detected": False,
        "report_date": None,
        "drifted_features": [],
        "total_features_checked": 0,
    }


@st.cache_data(ttl=300)
def load_slice_performance() -> pd.DataFrame:
    """Load per-slice performance table from GCS (CSV format)."""
    try:
        return pd.read_csv(SLICE_PERF_URI)
    except Exception as e:
        log.warning("Slice performance not available: %s", e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Predictions — with live scoring fallback
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner="Generating predictions...")
def load_predictions() -> pd.DataFrame:
    """Load or generate distress probability predictions.

    Tries in order:
      1. Batch scores parquet from GCS (Person 5's output)
      2. Local model + test split → live predict_proba
      3. GCS model + test split → live predict_proba
      4. Empty DataFrame (graceful failure)
    """
    # 1. Try batch scores
    try:
        df = pd.read_parquet(SCORES_URI)
        if "distress_probability" in df.columns:
            if df["distress_probability"].sum() > 0:
                log.info("Loaded batch scores: %d rows", len(df))
                return df
            log.warning("GCS scores are all zero — falling back to local scoring")
    except Exception:
        pass
    # 2/3. Live scoring fallback
    try:
        import tempfile

        from pandas.api.types import is_datetime64_any_dtype
        from xgboost import XGBClassifier

        if LOCAL_MODEL.exists() and LOCAL_TEST.exists():
            log.info("Scoring from local files...")
            model_path = LOCAL_MODEL
            test_df = pd.read_parquet(LOCAL_TEST)
        else:
            log.info("Scoring from GCS...")
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            model_path = Path(tempfile.gettempdir()) / "xgb_model.pkl"
            bucket.blob("models/xgb_model.pkl").download_to_filename(str(model_path))
            test_df = pd.read_parquet(f"gs://{GCS_BUCKET}/splits/v1/test.parquet")

        model = XGBClassifier()
        model.load_model(str(model_path))

        label_col = "distress_label"
        if label_col not in test_df.columns:
            log.warning("No label column in test data")
            return pd.DataFrame()

        # Preserve identifiers
        id_cols = ["firm_id", "fiscal_year", "fiscal_period"]
        ids = test_df[id_cols].copy()

        # Prepare features — drop non-numeric columns
        drop_cols = [label_col] + [
            c for c in test_df.columns
            if c in id_cols
            or test_df[c].dtype == "object"
            or is_datetime64_any_dtype(test_df[c])
        ]
        features = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
        features = pd.get_dummies(features, dummy_na=True)
        # Align to model's expected columns
        trained_cols = list(model.get_booster().feature_names or [])
        if trained_cols:
            features = features.reindex(columns=trained_cols, fill_value=0)

        probas = model.predict_proba(features)[:, 1]

        result = ids.copy()
        result["distress_probability"] = probas
        result["distress_label"] = test_df[label_col].values

        log.info("Live scoring complete: %d rows, mean prob=%.3f", len(result), probas.mean())
        return result

    except Exception as e:
        log.warning("Live scoring failed: %s", e)
        log.info(
            "To enable predictions, place xgb_model.pkl in artifacts/models/ "
            "and test.parquet in artifacts/splits/"
        )
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Company name mapping
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner="Loading company names...")
def load_company_map() -> pd.DataFrame:
    """Load company name/ticker/CIK mapping.

    Tries local files first, then GCS.
    Returns DataFrame with columns: firm_id, ticker, name.
    """
    empty = pd.DataFrame(columns=["firm_id", "ticker", "name"])
    try:
        if LOCAL_COMPANY_NAMES.exists():
            df = pd.read_csv(LOCAL_COMPANY_NAMES, dtype={"cik": str})
        elif LOCAL_COMPANY_REF.exists():
            df = pd.read_csv(LOCAL_COMPANY_REF)
            df["cik"] = df["cik"].astype(str)
            df["name"] = df["ticker"]
        else:
            # Try GCS
            try:
                df = pd.read_csv(f"gs://{GCS_BUCKET}/reference/company_names.csv", dtype={"cik": str})
            except Exception:
                try:
                    df = pd.read_csv(f"gs://{GCS_BUCKET}/reference/companies.csv")
                    df["cik"] = df["cik"].astype(str)
                    df["name"] = df["ticker"]
                except Exception:
                    return empty
        df["firm_id"] = df["cik"].str.zfill(10)
        log.info("Loaded %d company mappings", len(df))
        return df[["firm_id", "ticker", "name"]]
    except Exception as e:
        log.warning("Company map not available: %s", e)
        return empty

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_company_history(panel: pd.DataFrame, firm_id: str) -> pd.DataFrame:
    """Get all quarterly rows for a company, sorted chronologically."""
    if panel.empty:
        return pd.DataFrame()
    return panel[panel["firm_id"] == firm_id].sort_values(["fiscal_year", "fiscal_period"]).copy()


def get_shap_for_company(
    shap_df: pd.DataFrame,
    firm_id: str,
    fiscal_year: int | None = None,
) -> pd.DataFrame:
    """Get SHAP rows for a company, optionally filtered by year."""
    if shap_df.empty or "firm_id" not in shap_df.columns:
        return pd.DataFrame()
    mask = shap_df["firm_id"] == firm_id
    if fiscal_year is not None and "fiscal_year" in shap_df.columns:
        mask = mask & (shap_df["fiscal_year"] == fiscal_year)
    return shap_df[mask].copy()
