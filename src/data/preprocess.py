"""
Preprocessing module for SEC XBRL and FRED raw data.

Reads partitioned parquet files from local raw layer
(synced from GCS via gsutil), applies formatting and basic
fixes, and writes interim parquet files locally + uploads to GCS.

Raw layer contract (local mirror of GCS):
  SEC:  data/raw/sec_xbrl/cik=XXXXXXXXXX/data.parquet
  FRED: data/raw/fred/series_id=SERIES.parquet

Interim outputs:
  data/interim/sec_xbrl_long.parquet
  data/interim/fred_timeseries.parquet
  data/interim/preprocess_report.json

  Uploaded to:
  gs://{BUCKET}/interim/sec_xbrl_long.parquet
  gs://{BUCKET}/interim/fred_timeseries.parquet
  gs://{BUCKET}/interim/preprocess_report.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUCKET_NAME = os.getenv("GCP_BUCKET_RAW", "financial-distress-data")

LOCAL_SEC_RAW = Path(os.getenv("LOCAL_SEC_RAW", "data/raw/sec_xbrl"))
LOCAL_FRED_RAW = Path(os.getenv("LOCAL_FRED_RAW", "data/raw/fred"))
LOCAL_OUT_DIR = Path(os.getenv("LOCAL_OUT_DIR", "data/interim"))

GCS_SEC_OUT = os.getenv("GCS_SEC_OUT", "interim/sec_xbrl_long.parquet")
GCS_FRED_OUT = os.getenv("GCS_FRED_OUT", "interim/fred_timeseries.parquet")
GCS_REPORT_OUT = os.getenv("GCS_REPORT_OUT", "interim/preprocess_report.json")

VALID_FISCAL_PERIODS = {"Q1", "Q2", "Q3", "Q4", "FY"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upload helper (uses gsutil to avoid SDK billing issues)
# ---------------------------------------------------------------------------

def upload_to_gcs(local_path: Path, bucket: str, gcs_path: str) -> None:
    """Upload a local file to GCS using gsutil."""
    dest = f"gs://{bucket}/{gcs_path}"
    try:
        subprocess.run(
            ["gsutil", "cp", str(local_path), dest],
            check=True, capture_output=True, text=True,
        )
        log.info("Uploaded -> %s", dest)
    except FileNotFoundError:
        log.warning("gsutil not found; skipping upload of %s", dest)
    except subprocess.CalledProcessError as e:
        log.warning("gsutil upload failed for %s: %s", dest, e.stderr.strip())


# ---------------------------------------------------------------------------
# SEC preprocessing
# ---------------------------------------------------------------------------

def load_sec_raw(raw_dir: Path) -> pd.DataFrame:
    """Load all SEC XBRL partitioned parquet files from local directory."""
    parquet_files = sorted(raw_dir.glob("cik=*/data.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No SEC parquet files found under {raw_dir}")

    log.info("Found %d SEC parquet files", len(parquet_files))
    frames: list[pd.DataFrame] = []
    for i, fp in enumerate(parquet_files):
        if (i + 1) % 200 == 0:
            log.info("  Reading SEC file %d / %d ...", i + 1, len(parquet_files))

        df = pd.read_parquet(fp)

        # Extract CIK from directory name (cik=0000001750)
        m = re.search(r"cik=(\d+)", str(fp))
        cik = m.group(1).zfill(10) if m else None
        if cik is None:
            log.warning("Skipping file with unparseable CIK: %s", fp)
            continue

        # Always set cik from directory to ensure consistency
        df["cik"] = cik
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

def load_fred_raw(raw_dir: Path) -> pd.DataFrame:
    """Load all FRED partitioned parquet files from local directory."""
    parquet_files = sorted(raw_dir.glob("series_id=*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No FRED parquet files found under {raw_dir}")

    log.info("Found %d FRED parquet files", len(parquet_files))
    frames: list[pd.DataFrame] = []
    for fp in parquet_files:
        m = re.search(r"series_id=([^/]+)\.parquet", fp.name)
        if m is None:
            log.warning("Skipping file with unparseable series_id: %s", fp)
            continue
        sid = m.group(1)
        df = pd.read_parquet(fp)
        df["series_id"] = sid
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def preprocess_fred(df: pd.DataFrame) -> pd.DataFrame:
    """Apply formatting and basic fixes to FRED data.

    Rules:
      1. Standardize columns
      2. Parse date, snap to quarter-end
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
    sec_raw_dir: Path = LOCAL_SEC_RAW,
    fred_raw_dir: Path = LOCAL_FRED_RAW,
    out_dir: Path = LOCAL_OUT_DIR,
    bucket: str = BUCKET_NAME,
    sec_gcs_out: str = GCS_SEC_OUT,
    fred_gcs_out: str = GCS_FRED_OUT,
    report_gcs_out: str = GCS_REPORT_OUT,
) -> dict:
    """Execute the full preprocessing pipeline.

    Returns the validation report dict (useful for Airflow XCom).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- SEC ---
    log.info("Loading SEC raw data from %s ...", sec_raw_dir)
    sec_raw = load_sec_raw(sec_raw_dir)
    log.info("SEC raw rows: %d", len(sec_raw))

    sec = preprocess_sec(sec_raw)
    log.info("SEC after preprocessing: %d rows, %d unique CIKs",
             len(sec), sec["cik"].nunique())

    sec_local = out_dir / "sec_xbrl_long.parquet"
    sec.to_parquet(sec_local, index=False)
    log.info("Saved SEC interim: %s", sec_local)

    # --- FRED ---
    log.info("Loading FRED raw data from %s ...", fred_raw_dir)
    fred_raw = load_fred_raw(fred_raw_dir)
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

    # --- Upload to GCS ---
    upload_to_gcs(sec_local, bucket, sec_gcs_out)
    upload_to_gcs(fred_local, bucket, fred_gcs_out)
    upload_to_gcs(report_local, bucket, report_gcs_out)

    log.info("Preprocessing complete.")
    return report


def main() -> None:
    """CLI entry point."""
    run_preprocessing()


if __name__ == "__main__":
    main()