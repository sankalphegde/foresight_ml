"""
Preprocessing module for SEC XBRL and FRED raw data.

Reads partitioned parquet files from GCS raw layer,
applies formatting and basic fixes, and writes interim
parquet files back to GCS.

Raw layer contract:
  SEC:  gs://{BUCKET}/raw/sec_xbrl/cik=XXXXXXXXXX/data.parquet
  FRED: gs://{BUCKET}/raw/fred/series_id=SERIES.parquet

Interim outputs:
  gs://{BUCKET}/interim/sec_xbrl_long.parquet
  gs://{BUCKET}/interim/fred_timeseries.parquet
  gs://{BUCKET}/interim/preprocess_report.json
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
from google.cloud import storage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUCKET_NAME = os.getenv("GCP_BUCKET_RAW", "financial-distress-data")

GCS_SEC_RAW_PREFIX = os.getenv("GCS_SEC_RAW_PREFIX", "raw/sec_xbrl/")
GCS_FRED_RAW_PREFIX = os.getenv("GCS_FRED_RAW_PREFIX", "raw/fred/")

GCS_SEC_OUT = os.getenv("GCS_SEC_OUT", "interim/sec_xbrl_long.parquet")
GCS_FRED_OUT = os.getenv("GCS_FRED_OUT", "interim/fred_timeseries.parquet")
GCS_REPORT_OUT = os.getenv("GCS_REPORT_OUT", "interim/preprocess_report.json")

LOCAL_OUT_DIR = Path(os.getenv("LOCAL_OUT_DIR", "data/interim"))

VALID_FISCAL_PERIODS = {"Q1", "Q2", "Q3", "Q4", "FY"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _gcs_client() -> storage.Client:
    return storage.Client()


def list_blobs(bucket: str, prefix: str, suffix: str = ".parquet") -> list[str]:
    """Return blob names under *prefix* that end with *suffix*."""
    client = _gcs_client()
    blobs = client.list_blobs(bucket, prefix=prefix)
    return [b.name for b in blobs if b.name.endswith(suffix)]


def read_parquet_from_gcs(bucket: str, blob_name: str) -> pd.DataFrame:
    """Read a single parquet file from GCS into a DataFrame."""
    import io
    client = _gcs_client()
    blob = client.bucket(bucket).blob(blob_name)
    data = blob.download_as_bytes()
    return pd.read_parquet(io.BytesIO(data))


def upload_to_gcs(local_path: Path, bucket: str, gcs_path: str) -> None:
    """Upload a local file to GCS."""
    client = _gcs_client()
    blob = client.bucket(bucket).blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    log.info("Uploaded -> gs://%s/%s", bucket, gcs_path)


# ---------------------------------------------------------------------------
# SEC preprocessing
# ---------------------------------------------------------------------------

def _extract_cik_from_blob(blob_name: str) -> str | None:
    """Extract 10-digit CIK from blob path like raw/sec_xbrl/cik=0000001750/data.parquet."""
    m = re.search(r"cik=(\d+)", blob_name)
    return m.group(1).zfill(10) if m else None


def load_sec_raw(bucket: str, prefix: str) -> pd.DataFrame:
    """Load all SEC XBRL partitioned parquet files into one DataFrame."""
    blobs = list_blobs(bucket, prefix)
    if not blobs:
        raise FileNotFoundError(f"No SEC parquet files found under gs://{bucket}/{prefix}")

    log.info("Found %d SEC parquet files", len(blobs))
    frames: list[pd.DataFrame] = []
    for i, blob_name in enumerate(blobs):
        if (i + 1) % 200 == 0:
            log.info("  Reading SEC file %d / %d ...", i + 1, len(blobs))
        df = read_parquet_from_gcs(bucket, blob_name)
        # The hive partition column 'cik' may or may not be auto-included
        if "cik" not in df.columns:
            cik = _extract_cik_from_blob(blob_name)
            if cik is None:
                log.warning("Skipping blob with unparseable CIK: %s", blob_name)
                continue
            df["cik"] = cik
        # Normalize cik to string to avoid type conflicts across files
        if "cik" in df.columns:
            df["cik"] = df["cik"].astype(str)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def preprocess_sec(df: pd.DataFrame) -> pd.DataFrame:
    """Apply formatting and basic fixes to raw SEC long-format data.

    Rules:
      1. Standardize column names (lowercase, strip)
      2. Zero-pad CIK to 10 digits
      3. Cast fiscal_year to int, drop nulls
      4. Keep only valid fiscal_period values (Q1-Q4, FY)
      5. Ensure quarter_key is consistent (<fiscal_year>_<fiscal_period>)
      6. Parse date columns
      7. Deduplicate by (cik, fiscal_year, fiscal_period, tag),
         keeping the row with the latest filed_date
    """
    # 1. Standardize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # 2. CIK zero-pad
    df["cik"] = df["cik"].astype(str).str.zfill(10)

    # 3. fiscal_year: drop rows where it's null
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["fiscal_year"])
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    n_dropped_fy = n_before - len(df)
    if n_dropped_fy:
        log.info("Dropped %d rows with null fiscal_year", n_dropped_fy)

    # 4. Filter valid fiscal_period
    df["fiscal_period"] = df["fiscal_period"].astype(str).str.strip().str.upper()
    n_before = len(df)
    df = df[df["fiscal_period"].isin(VALID_FISCAL_PERIODS)]
    n_dropped_fp = n_before - len(df)
    if n_dropped_fp:
        log.info("Dropped %d rows with invalid fiscal_period", n_dropped_fp)

    # 5. Rebuild quarter_key for consistency
    df["quarter_key"] = df["fiscal_year"].astype(str) + "_" + df["fiscal_period"]

    # 6. Parse dates
    for col in ["filed_date", "end_date", "start_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 7. Deduplicate: keep latest filed_date per (cik, fiscal_year, fiscal_period, tag)
    n_before = len(df)
    sort_col = "filed_date" if "filed_date" in df.columns else None
    if sort_col is not None:
        df = df.sort_values(sort_col, ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["cik", "fiscal_year", "fiscal_period", "tag"], keep="first")
    n_deduped = n_before - len(df)
    if n_deduped:
        log.info("Removed %d duplicate rows", n_deduped)

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# FRED preprocessing
# ---------------------------------------------------------------------------

def _extract_series_id_from_blob(blob_name: str) -> str | None:
    """Extract series_id from blob like raw/fred/series_id=CPIAUCSL.parquet."""
    m = re.search(r"series_id=([^/]+)\.parquet", blob_name)
    return m.group(1) if m else None


def load_fred_raw(bucket: str, prefix: str) -> pd.DataFrame:
    """Load all FRED partitioned parquet files into one DataFrame."""
    blobs = list_blobs(bucket, prefix)
    if not blobs:
        raise FileNotFoundError(f"No FRED parquet files found under gs://{bucket}/{prefix}")

    log.info("Found %d FRED parquet files", len(blobs))
    frames: list[pd.DataFrame] = []
    for blob_name in blobs:
        sid = _extract_series_id_from_blob(blob_name)
        if sid is None:
            log.warning("Skipping blob with unparseable series_id: %s", blob_name)
            continue
        df = read_parquet_from_gcs(bucket, blob_name)
        df["series_id"] = sid
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def preprocess_fred(df: pd.DataFrame) -> pd.DataFrame:
    """Apply formatting and basic fixes to FRED data.

    Rules:
      1. Standardize columns
      2. Parse date, ensure it is quarter-end
      3. Cast value to float
      4. Deduplicate by (series_id, date)
      5. Sort ascending by series_id, date
    """
    # 1. Standardize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # 2. Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["date"])
    n_dropped = n_before - len(df)
    if n_dropped:
        log.info("Dropped %d FRED rows with null date", n_dropped)

    # Snap to quarter-end
    df["date"] = df["date"] + pd.offsets.QuarterEnd(0)

    # 3. Value
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 4. Dedup
    n_before = len(df)
    df = df.drop_duplicates(subset=["series_id", "date"], keep="last")
    n_deduped = n_before - len(df)
    if n_deduped:
        log.info("Removed %d duplicate FRED rows", n_deduped)

    # 5. Sort
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def build_report(sec: pd.DataFrame, fred: pd.DataFrame) -> dict:
    """Build a JSON-serializable validation report."""
    def _col_stats(df: pd.DataFrame) -> dict:
        stats: dict = {
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "null_counts": {c: int(v) for c, v in df.isna().sum().items()},
        }
        num_cols = df.select_dtypes(include="number").columns
        stats["numeric_ranges"] = {
            c: {"min": float(df[c].min()), "max": float(df[c].max())}
            for c in num_cols if len(df) > 0
        }
        return stats

    return {
        "sec": _col_stats(sec),
        "fred": _col_stats(fred),
        "unique_ciks": int(sec["cik"].nunique()) if "cik" in sec.columns else 0,
        "unique_fred_series": int(fred["series_id"].nunique()) if "series_id" in fred.columns else 0,
    }


# ---------------------------------------------------------------------------
# Main entry point (Airflow-friendly)
# ---------------------------------------------------------------------------

def run_preprocessing(
    bucket: str = BUCKET_NAME,
    sec_prefix: str = GCS_SEC_RAW_PREFIX,
    fred_prefix: str = GCS_FRED_RAW_PREFIX,
    sec_out: str = GCS_SEC_OUT,
    fred_out: str = GCS_FRED_OUT,
    report_out: str = GCS_REPORT_OUT,
) -> dict:
    """Execute the full preprocessing pipeline.

    Returns the validation report dict (useful for Airflow XCom).
    """
    out_dir = LOCAL_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- SEC ---
    log.info("Loading SEC raw data from gs://%s/%s ...", bucket, sec_prefix)
    sec_raw = load_sec_raw(bucket, sec_prefix)
    log.info("SEC raw rows: %d", len(sec_raw))

    sec = preprocess_sec(sec_raw)
    log.info("SEC after preprocessing: %d rows, %d unique CIKs",
             len(sec), sec["cik"].nunique())

    sec_local = out_dir / "sec_xbrl_long.parquet"
    sec.to_parquet(sec_local, index=False)
    log.info("Saved SEC interim: %s", sec_local)

    # --- FRED ---
    log.info("Loading FRED raw data from gs://%s/%s ...", bucket, fred_prefix)
    fred_raw = load_fred_raw(bucket, fred_prefix)
    log.info("FRED raw rows: %d", len(fred_raw))

    fred = preprocess_fred(fred_raw)
    log.info("FRED after preprocessing: %d rows, %d series",
             len(fred), fred["series_id"].nunique())

    fred_local = out_dir / "fred_timeseries.parquet"
    fred.to_parquet(fred_local, index=False)
    log.info("Saved FRED interim: %s", fred_local)

    # --- Report ---
    report = build_report(sec, fred)
    report_local = out_dir / "preprocess_report.json"
    with open(report_local, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Saved report: %s", report_local)

    # --- Upload ---
    upload_to_gcs(sec_local, bucket, sec_out)
    upload_to_gcs(fred_local, bucket, fred_out)
    upload_to_gcs(report_local, bucket, report_out)

    log.info("Preprocessing complete.")
    return report


def main() -> None:
    """CLI entry point."""
    run_preprocessing()


if __name__ == "__main__":
    main()