from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from google.cloud import storage  # type: ignore[attr-defined]

BUCKET_NAME = os.getenv("GCP_BUCKET_RAW", "financial-distress-data")
GCS_OUT_PATH = os.getenv("GCS_PREPROCESS_OUT", "interim/panel_base.parquet")
GCS_REPORT_PATH = os.getenv("GCS_PREPROCESS_REPORT_OUT", "interim/preprocess_report.json")

# Optional: SEC XBRL long facts (many parquet files)
GCS_XBRL_LONG_OUT_PATH = os.getenv("GCS_XBRL_LONG_OUT", "interim/sec_xbrl_long.parquet")
XBRL_LONG_LOCAL_DIR = os.getenv("XBRL_LONG_LOCAL_DIR", "data/raw/sec_xbrl_long")


def read_sec_jsonl(path: str) -> pd.DataFrame:
    rows = []
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
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # convert all columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def upload_to_gcs(local_path: Path, bucket_name: str, gcs_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"Uploaded -> gs://{bucket_name}/{gcs_path}")


def main() -> None:
    sec_path = "data/raw/sec/filings.jsonl"
    fred_path = "data/raw/fred/indicators.csv"

    # Check if required files exist
    if not Path(sec_path).exists():
        raise FileNotFoundError(f"SEC data file not found: {sec_path}")
    if not Path(fred_path).exists():
        raise FileNotFoundError(f"FRED data file not found: {fred_path}")

    print("Reading SEC...")
    sec = read_sec_jsonl(sec_path)

    print("Reading FRED...")
    fred = read_fred_csv(fred_path)

    # FRED has multiple rows but no date column.
    # Use first row as latest snapshot
    latest_fred = fred.iloc[0]

    for col in fred.columns:
        sec[col] = latest_fred[col]

    # Save interim output
    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "panel_base.parquet"

    sec.to_parquet(out_path, index=False)

    # --- VALIDATION REPORT ---
    print("Creating validation report...")

    report = {
        "row_count": int(len(sec)),
        "columns": list(sec.columns),
        "null_counts": sec.isna().sum().to_dict(),
    }

    # numeric ranges
    numeric_cols = sec.select_dtypes(include="number").columns
    report["numeric_ranges"] = {
        col: {
            "min": float(sec[col].min()) if len(sec) else None,
            "max": float(sec[col].max()) if len(sec) else None,
        }
        for col in numeric_cols
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

        # Upload xbrl long parquet
        upload_to_gcs(xbrl_out_path, BUCKET_NAME, GCS_XBRL_LONG_OUT_PATH)
    else:
        print("No local SEC XBRL long parquet files found. Skipping XBRL long output.")

    # Upload main outputs
    upload_to_gcs(out_path, BUCKET_NAME, GCS_OUT_PATH)
    upload_to_gcs(report_path, BUCKET_NAME, GCS_REPORT_PATH)


if __name__ == "__main__":
    main()
