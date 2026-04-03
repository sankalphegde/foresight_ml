"""GCS data loader for the Streamlit dashboard.

Loads scores, SHAP values, labeled panel, manifest, drift reports,
and model metadata from GCS. All loaders are cached with a 5-minute TTL
so the dashboard stays responsive.

Gracefully returns empty DataFrames / default dicts when files don't exist yet.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GCS paths
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
# Loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner="Loading scores...")
def load_scores() -> pd.DataFrame:
    """Load batch inference scores from GCS."""
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
    """Load manifest.json from GCS."""
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
    return {"baseline_val_roc": 0.0, "best_params": {}, "test_roc_auc": 0.0}


@st.cache_data(ttl=300)
def load_drift_summary() -> dict:
    """Load latest drift monitoring summary from GCS."""
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
        df = pd.read_csv(SLICE_PERF_URI)
        return df
    except Exception as e:
        log.warning("Slice performance not available: %s", e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Predictions — live scoring fallback
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner="Generating predictions...")
def load_predictions() -> pd.DataFrame:
    """Load or generate distress probability predictions.

    Priority:
      1. Load scores parquet from GCS (Person 5's batch inference output)
      2. Fall back: load model + test data, run predict_proba live

    Returns DataFrame with firm_id, fiscal_year, fiscal_period, distress_probability.
    """
    # Try scores parquet first
    try:
        df = pd.read_parquet(SCORES_URI)
        if "distress_probability" in df.columns:
            log.info("Loaded batch scores: %d rows", len(df))
            return df
    except Exception:
        pass

    # Fall back: score using model + test data (local first, then GCS)
    try:
        import tempfile
        from pathlib import Path

        from pandas.api.types import is_datetime64_any_dtype
        from xgboost import XGBClassifier

        # Try local files first, then GCS
        local_model = Path("artifacts/models/xgb_model.pkl")
        local_test = Path("artifacts/splits/test.parquet")

        if local_model.exists() and local_test.exists():
            log.info("Scoring from local files...")
            model_path = local_model
            test_df = pd.read_parquet(local_test)
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
            return pd.DataFrame()

        # Preserve identifiers
        id_cols = ["firm_id", "fiscal_year", "fiscal_period"]
        ids = test_df[id_cols].copy()

        # Prepare features
        features = test_df.drop(columns=[label_col])
        for col in features.columns:
            if is_datetime64_any_dtype(features[col]):
                features[col] = features[col].astype("int64")
        features = pd.get_dummies(features, dummy_na=True)

        # Align to model's expected columns
        trained_cols = model.get_booster().feature_names
        if trained_cols:
            features = features.reindex(columns=trained_cols, fill_value=0)

        # Predict
        probas = model.predict_proba(features)[:, 1]

        result = ids.copy()
        result["distress_probability"] = probas
        result["distress_label"] = test_df[label_col].values

        log.info("Live scoring complete: %d rows, mean prob=%.3f", len(result), probas.mean())
        return result

    except Exception as e:
        log.warning("Live scoring failed: %s", e)
        return pd.DataFrame()
        import tempfile
        from pathlib import Path

        from google.cloud import storage
        from pandas.api.types import is_datetime64_any_dtype
        from xgboost import XGBClassifier

        log.info("Scoring live using model from GCS...")

        # Download model
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        model_path = Path(tempfile.gettempdir()) / "xgb_model.pkl"
        bucket.blob("models/xgb_model.pkl").download_to_filename(str(model_path))

        model = XGBClassifier()
        model.load_model(str(model_path))

        # Load test split
        test_df = pd.read_parquet(f"gs://{GCS_BUCKET}/splits/v1/test.parquet")

        label_col = "distress_label"
        if label_col not in test_df.columns:
            return pd.DataFrame()

        # Preserve identifiers
        id_cols = ["firm_id", "fiscal_year", "fiscal_period"]
        ids = test_df[id_cols].copy()

        # Prepare features
        features = test_df.drop(columns=[label_col])
        for col in features.columns:
            if is_datetime64_any_dtype(features[col]):
                features[col] = features[col].astype("int64")
        features = pd.get_dummies(features, dummy_na=True)

        # Align to model's expected columns
        trained_cols = model.get_booster().feature_names
        if trained_cols:
            features = features.reindex(columns=trained_cols, fill_value=0)

        # Predict
        probas = model.predict_proba(features)[:, 1]

        result = ids.copy()
        result["distress_probability"] = probas
        result["distress_label"] = test_df[label_col].values

        log.info("Live scoring complete: %d rows, mean prob=%.3f", len(result), probas.mean())
        return result

    except Exception as e:
        log.warning("Live scoring failed: %s", e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Derived data helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading company names...")
def load_company_map() -> dict[str, str]:
    """Load ticker → firm_id mapping from reference/companies.csv.

    Returns dict like {"AAPL": "0000320193", "MSFT": "0000789019"}.
    """
    try:
        from pathlib import Path

        local = Path("artifacts/reference/companies.csv")
        if local.exists():
            df = pd.read_csv(local)
        else:
            df = pd.read_csv(f"gs://{GCS_BUCKET}/reference/companies.csv")

        # Zero-pad CIK to 10 digits to match panel's firm_id format
        df["firm_id"] = df["cik"].astype(str).str.zfill(10)
        mapping = dict(zip(df["ticker"], df["firm_id"]))
        log.info("Loaded %d ticker mappings", len(mapping))
        return mapping
    except Exception as e:
        log.warning("Company map not available: %s", e)
        return {}


def reverse_company_map(company_map: dict[str, str]) -> dict[str, str]:
    """Reverse the ticker→firm_id map to firm_id→ticker."""
    return {v: k for k, v in company_map.items()}

@st.cache_data(ttl=3600, show_spinner="Loading company names...")
def load_company_map() -> pd.DataFrame:
    """Load company name/ticker/CIK mapping.

    Tries local SEC company names file first, falls back to reference CSV.
    Returns DataFrame with columns: firm_id, ticker, name.
    """
    from pathlib import Path

    try:
        local = Path("artifacts/reference/company_names.csv")
        if local.exists():
            df = pd.read_csv(local, dtype={"cik": str})
        else:
            # Fall back to reference/companies.csv (ticker + cik only)
            ref = Path("artifacts/reference/companies.csv")
            if ref.exists():
                df = pd.read_csv(ref)
                df["cik"] = df["cik"].astype(str)
                df["name"] = df["ticker"]  # use ticker as name fallback
            else:
                return pd.DataFrame(columns=["firm_id", "ticker", "name"])

        df["firm_id"] = df["cik"].str.zfill(10)
        log.info("Loaded %d company mappings", len(df))
        return df[["firm_id", "ticker", "name"]]
    except Exception as e:
        log.warning("Company map not available: %s", e)
        return pd.DataFrame(columns=["firm_id", "ticker", "name"])

def get_company_list(panel: pd.DataFrame) -> pd.DataFrame:
    """Get unique company list with latest data for search/filter."""
    if panel.empty:
        return pd.DataFrame()

    panel = panel.sort_values(["firm_id", "fiscal_year", "fiscal_period"])
    latest = panel.groupby("firm_id").last().reset_index()

    cols = ["firm_id", "fiscal_year", "fiscal_period"]
    optional = [
        "total_assets",
        "net_income",
        "distress_label",
        "company_size_bucket",
        "sector_proxy",
    ]
    for c in optional:
        if c in latest.columns:
            cols.append(c)

    return latest[cols]


def get_company_history(panel: pd.DataFrame, firm_id: str) -> pd.DataFrame:
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
