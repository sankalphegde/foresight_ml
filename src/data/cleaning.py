from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import storage

BUCKET_NAME = os.getenv("GCP_BUCKET_RAW", "financial-distress-data")
GCS_OUT_PATH = os.getenv("GCS_CLEANED_OUT", "processed/cleaned_panel.parquet")
GCS_REPORT_PATH = os.getenv("GCS_CLEANING_REPORT_OUT", "processed/cleaning_report.json")

INPUT_PATH = os.getenv("CLEANING_INPUT_PATH", "data/interim/panel_base.parquet")
OUTPUT_DIR = os.getenv("CLEANING_OUTPUT_DIR", "data/processed")

REQUIRED_COLUMNS = ("cik", "filing_date")

ACCOUNTING_NONNEGATIVE_COLS = (
    "assets",
    "total_assets",
    "liabilities",
    "total_liabilities",
    "revenue",
    "net_sales",
    "cash",
    "cash_and_cash_equivalents",
    "operating_cash_flow",
)

BALANCE_SHEET_ASSETS = ("assets", "total_assets")
BALANCE_SHEET_LIABILITIES = ("liabilities", "total_liabilities")
BALANCE_SHEET_EQUITY = ("equity", "stockholders_equity", "total_equity")


def upload_to_gcs(local_path: Path, bucket_name: str, gcs_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"Uploaded -> gs://{bucket_name}/{gcs_path}")


def _mode_or_unknown(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    if mode.empty:
        return "UNKNOWN"
    value = mode.iloc[0]
    if pd.isna(value):
        return "UNKNOWN"
    return str(value)


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def run_accounting_checks(df: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {"skipped": [], "violations": {}}

    assets_col = _first_existing_column(df, BALANCE_SHEET_ASSETS)
    liab_col = _first_existing_column(df, BALANCE_SHEET_LIABILITIES)
    equity_col = _first_existing_column(df, BALANCE_SHEET_EQUITY)

    if assets_col and liab_col and equity_col:
        assets = pd.to_numeric(df[assets_col], errors="coerce")
        liabilities = pd.to_numeric(df[liab_col], errors="coerce")
        equity = pd.to_numeric(df[equity_col], errors="coerce")
        diff = (assets - (liabilities + equity)).abs()
        tol = assets.abs() * 0.01
        tol = tol.fillna(0)
        violations = diff > tol
        checks["violations"]["balance_sheet_identity"] = {
            "assets_col": assets_col,
            "liabilities_col": liab_col,
            "equity_col": equity_col,
            "count": int(violations.sum()),
        }
    else:
        checks["skipped"].append("balance_sheet_identity_missing_columns")

    nonneg_cols = [c for c in ACCOUNTING_NONNEGATIVE_COLS if c in df.columns]
    if nonneg_cols:
        violations = {}
        for col in nonneg_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            violations[col] = int((series < 0).sum())
        checks["violations"]["non_negative"] = violations
    else:
        checks["skipped"].append("non_negative_missing_columns")

    if assets_col and liab_col:
        assets = pd.to_numeric(df[assets_col], errors="coerce")
        liabilities = pd.to_numeric(df[liab_col], errors="coerce")
        ratio = liabilities / assets.replace(0, pd.NA)
        invalid = (ratio < 0) | (ratio > 5)
        checks["violations"]["debt_to_assets_bounds"] = int(invalid.sum())
    else:
        checks["skipped"].append("debt_to_assets_missing_columns")

    if "current_assets" in df.columns and "current_liabilities" in df.columns:
        ca = pd.to_numeric(df["current_assets"], errors="coerce")
        cl = pd.to_numeric(df["current_liabilities"], errors="coerce")
        ratio = ca / cl.replace(0, pd.NA)
        invalid = (ratio <= 0) | (ratio > 10)
        checks["violations"]["current_ratio_bounds"] = int(invalid.sum())
    else:
        checks["skipped"].append("current_ratio_missing_columns")

    return checks


def clean_and_impute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    report: dict[str, Any] = {
        "row_count_before": int(len(df)),
        "null_counts_before": df.isna().sum().to_dict(),
    }

    # Drop rows missing required identifiers
    before_drop = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    report["rows_dropped"] = int(before_drop - len(df))

    # Ensure datetime type for filing_date if present
    if "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

    imputation: dict[str, dict[str, Any]] = {}

    # Numeric imputation: median per cik, fallback to global median
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        before = int(df[col].isna().sum())
        if before == 0:
            continue
        group_median = df.groupby("cik")[col].transform("median")
        df[col] = df[col].fillna(group_median)
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)
        after = int(df[col].isna().sum())
        imputation[col] = {
            "strategy": "group_median_then_global_median",
            "filled": int(before - after),
        }

    # Categorical imputation: mode, fallback to UNKNOWN
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in categorical_cols:
        if col in REQUIRED_COLUMNS:
            continue
        before = int(df[col].isna().sum())
        if before == 0:
            continue
        fill_value = _mode_or_unknown(df[col])
        df[col] = df[col].fillna(fill_value)
        after = int(df[col].isna().sum())
        imputation[col] = {
            "strategy": "mode_or_unknown",
            "filled": int(before - after),
            "fill_value": fill_value,
        }

    report["row_count_after"] = int(len(df))
    report["null_counts_after"] = df.isna().sum().to_dict()
    report["imputation"] = imputation
    report["accounting_checks"] = run_accounting_checks(df)

    return df, report


def main() -> None:
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Interim dataset not found: {in_path}")

    df = pd.read_parquet(in_path)
    cleaned, report = clean_and_impute(df)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cleaned_panel.parquet"
    report_path = out_dir / "cleaning_report.json"

    cleaned.to_parquet(out_path, index=False)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved cleaned parquet:", out_path)
    print("Saved cleaning report:", report_path)
    print("Rows:", len(cleaned))
    print("Columns:", list(cleaned.columns))

    upload_to_gcs(out_path, BUCKET_NAME, GCS_OUT_PATH)
    upload_to_gcs(report_path, BUCKET_NAME, GCS_REPORT_PATH)


if __name__ == "__main__":
    main()
