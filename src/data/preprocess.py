"""Data preprocessing module for SEC and FRED data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from google.cloud import storage

BUCKET_NAME = os.getenv("GCP_BUCKET_RAW", "financial-distress-data")

# Main outputs
GCS_OUT_PATH = os.getenv("GCS_PREPROCESS_OUT", "interim/panel_base.parquet")
GCS_REPORT_PATH = os.getenv("GCS_PREPROCESS_REPORT_OUT", "interim/preprocess_report.json")

# Optional: SEC XBRL long facts (many parquet files)
GCS_XBRL_LONG_OUT_PATH = os.getenv("GCS_XBRL_LONG_OUT", "interim/sec_xbrl_long.parquet")
XBRL_LONG_LOCAL_DIR = os.getenv("XBRL_LONG_LOCAL_DIR", "data/raw/sec_xbrl_long")

# FRED macro history
GCS_FRED_TS_OUT_PATH = os.getenv("GCS_FRED_TS_OUT", "interim/fred_timeseries.parquet")


def read_sec_jsonl(path: str) -> pd.DataFrame:
    """Read SEC filings from JSONL file and return standardized DataFrame."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    # Standardize
    df.columns = [c.strip().lower() for c in df.columns]
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

    # Quarter label
    df["quarter"] = df["filing_date"].dt.to_period("Q").astype(str)

    # Single date column
    df["date"] = df["filing_date"]

    # Remove duplicates
    df = df.drop_duplicates(subset=["cik", "accession_number"])

    return df


def read_fred_csv(path: str) -> pd.DataFrame:
    """Read FRED economic indicators from CSV file and return DataFrame."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Convert all columns to numeric if possible
    for c in df.columns:
        # if there is a date column in the future, keep it parseable
        if c == "date":
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def upload_to_gcs(local_path: Path, bucket_name: str, gcs_path: str) -> None:
    """Upload local file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"Uploaded -> gs://{bucket_name}/{gcs_path}")


def main() -> None:
    """Main preprocessing entry point."""
    sec_path = "data/raw/sec/filings.jsonl"
    fred_path = "data/raw/fred/indicators.csv"

    # Check if required files exist
    if not Path(sec_path).exists():
        raise FileNotFoundError(f"SEC data file not found: {sec_path}")
    if not Path(fred_path).exists():
        raise FileNotFoundError(f"FRED data file not found: {fred_path}")

    # Output dir
    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading SEC...")
    sec = read_sec_jsonl(sec_path)

    print("Reading FRED...")
    fred = read_fred_csv(fred_path)

    # --- FRED TIME SERIES OUTPUT (save full macro history) ---
    fred_ts = fred.copy()

    # If a date column exists, normalize it (optional)
    if "date" in fred_ts.columns:
        fred_ts["date"] = pd.to_datetime(fred_ts["date"], errors="coerce")
        fred_ts = fred_ts.sort_values("date").reset_index(drop=True)

    fred_ts_out_path = out_dir / "fred_timeseries.parquet"
    fred_ts.to_parquet(fred_ts_out_path, index=False)
    print("Saved FRED timeseries parquet:", fred_ts_out_path)
    print("FRED timeseries rows:", len(fred_ts))

    upload_to_gcs(fred_ts_out_path, BUCKET_NAME, GCS_FRED_TS_OUT_PATH)

    # --- FRED SNAPSHOT (attach latest macro values onto each SEC filing row) ---
    # Current raw indicators.csv in your bucket has no date column.
    # We treat row 0 as the "latest snapshot" (consistent with your existing approach).
    latest_fred = fred.iloc[0]

    for col in fred.columns:
        # avoid overwriting SEC's own date column if FRED ever adds one later
        if col == "date":
            continue
        sec[col] = latest_fred[col]

    # Save interim SEC+macro snapshot panel
    out_path = out_dir / "panel_base.parquet"
    sec.to_parquet(out_path, index=False)

    # --- VALIDATION REPORT ---
    print("Creating validation report...")

    report: dict = {
        "row_count": int(len(sec)),
        "columns": list(sec.columns),
        "null_counts": sec.isna().sum().to_dict(),
    }

    numeric_cols = sec.select_dtypes(include="number").columns
    report["numeric_ranges"] = {
        col: {
            "min": float(sec[col].min()) if len(sec) else None,
            "max": float(sec[col].max()) if len(sec) else None,
        }
        for col in numeric_cols
    }

    # Optional: include FRED TS shape info
    report["fred_timeseries"] = {
        "row_count": int(len(fred_ts)),
        "columns": list(fred_ts.columns),
    }

    report_path = out_dir / "preprocess_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved validation report:", report_path)
    print("Saved interim parquet:", out_path)
    print("Rows:", len(sec))
    print("Columns:", list(sec.columns))

    # --- OPTIONAL: SEC XBRL LONG (facts) ---
    xbrl_dir = Path(XBRL_LONG_LOCAL_DIR)
    xbrl_paths = sorted(xbrl_dir.glob("*.parquet")) if xbrl_dir.exists() else []

    if xbrl_paths:
        print(f"Reading SEC XBRL long facts from {xbrl_dir} ...")
        xbrl_df = pd.concat([pd.read_parquet(p) for p in xbrl_paths], ignore_index=True)

        xbrl_out_path = out_dir / "sec_xbrl_long.parquet"
        xbrl_df.to_parquet(xbrl_out_path, index=False)
        print("Saved XBRL long parquet:", xbrl_out_path)
        print("XBRL long rows:", len(xbrl_df))

        report["xbrl_long"] = {
            "row_count": int(len(xbrl_df)),
            "columns": list(xbrl_df.columns),
        }

        upload_to_gcs(xbrl_out_path, BUCKET_NAME, GCS_XBRL_LONG_OUT_PATH)
    else:
        print("No local SEC XBRL long parquet files found. Skipping XBRL long output.")

    # Upload main outputs
    upload_to_gcs(out_path, BUCKET_NAME, GCS_OUT_PATH)
    upload_to_gcs(report_path, BUCKET_NAME, GCS_REPORT_PATH)


if __name__ == "__main__":
    main()
