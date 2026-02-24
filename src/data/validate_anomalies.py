"""Validation and anomaly detection pipeline for processed panel data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

BUCKET_NAME = os.getenv("GCP_BUCKET_RAW", "financial-distress-data")
INPUT_PATH = os.getenv("VALIDATION_INPUT_PATH", "data/processed/cleaned_panel.parquet")
OUTPUT_DIR = os.getenv("VALIDATION_OUTPUT_DIR", "data/processed")
GCS_REPORT_PATH = os.getenv("GCS_VALIDATION_REPORT_OUT", "processed/validation_report.json")
GCS_ANOMALIES_PATH = os.getenv("GCS_ANOMALIES_OUT", "processed/anomalies.parquet")

REQUIRED_COLUMNS = ("cik", "filing_date", "ticker", "accession_number")


def upload_to_gcs(local_path: Path, bucket_name: str, gcs_path: str) -> None:
    """Upload a local file to a target GCS object path."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"Uploaded -> gs://{bucket_name}/{gcs_path}")


def detect_anomalies_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Detect row-level numeric outliers using the 1.5*IQR rule."""
    numeric_cols = list(df.select_dtypes(include="number").columns)
    if not numeric_cols:
        return df.iloc[0:0].copy(), {}

    flags = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    anomaly_counts: dict[str, int] = {}

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            flags[col] = False
            anomaly_counts[col] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        col_flags = (s < lower) | (s > upper)
        col_flags = col_flags.fillna(False)

        flags[col] = col_flags
        anomaly_counts[col] = int(col_flags.sum())

    row_flag = flags.any(axis=1)
    anomalies = df.loc[row_flag].copy()

    if not anomalies.empty:
        anomalies["anomaly_count"] = flags.loc[row_flag].sum(axis=1).astype(int)
        anomalies["anomaly_columns"] = flags.loc[row_flag].apply(
            lambda r: ",".join([c for c, v in r.items() if bool(v)]), axis=1
        )

    return anomalies, anomaly_counts


def build_validation_report(
    df: pd.DataFrame, anomalies: pd.DataFrame, anomaly_counts: dict[str, int]
) -> dict[str, Any]:
    """Build a JSON-serializable validation summary for orchestration."""
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    duplicate_count = 0
    if {"cik", "accession_number"}.issubset(df.columns):
        duplicate_count = int(df.duplicated(subset=["cik", "accession_number"]).sum())

    null_counts = {k: int(v) for k, v in df.isna().sum().to_dict().items()}
    row_count = int(len(df))
    null_rates = {k: (v / row_count if row_count else 0.0) for k, v in null_counts.items()}

    numeric_ranges: dict[str, dict[str, float | None]] = {}
    for col in df.select_dtypes(include="number").columns:
        s = pd.to_numeric(df[col], errors="coerce")
        mn = s.min()
        mx = s.max()
        numeric_ranges[col] = {
            "min": (float(mn) if pd.notna(mn) else None),
            "max": (float(mx) if pd.notna(mx) else None),
        }

    status = "pass"
    if missing_required or duplicate_count > 0:
        status = "fail"

    report: dict[str, Any] = {
        "status": status,
        "row_count": row_count,
        "anomaly_count": int(len(anomalies)),
        "required_columns": list(REQUIRED_COLUMNS),
        "missing_required_columns": missing_required,
        "duplicate_count_cik_accession_number": duplicate_count,
        "null_counts": null_counts,
        "null_rates": null_rates,
        "numeric_ranges": numeric_ranges,
        "anomalies_by_column": anomaly_counts,
    }
    return report


def validate_and_detect(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run all validation checks and anomaly detection for a dataframe."""
    anomalies, anomaly_counts = detect_anomalies_iqr(df)
    report = build_validation_report(df, anomalies, anomaly_counts)
    return anomalies, report


def main() -> None:
    """Execute validation pipeline and upload generated artifacts to GCS."""
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {in_path}")

    df = pd.read_parquet(in_path)
    anomalies, report = validate_and_detect(df)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "validation_report.json"
    anomalies_path = out_dir / "anomalies.parquet"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    anomalies.to_parquet(anomalies_path, index=False)

    print("Saved validation report:", report_path)
    print("Saved anomalies parquet:", anomalies_path)
    print("Status:", report["status"])
    print("Rows:", report["row_count"])
    print("Anomaly rows:", report["anomaly_count"])

    upload_to_gcs(report_path, BUCKET_NAME, GCS_REPORT_PATH)
    upload_to_gcs(anomalies_path, BUCKET_NAME, GCS_ANOMALIES_PATH)


if __name__ == "__main__":
    main()
