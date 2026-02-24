"""Unit tests for validation and anomaly detection pipeline."""

import sys
from pathlib import Path

import pandas as pd

# Ensure local project package is imported instead of similarly named site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validate_anomalies import validate_and_detect


def test_duplicate_detection_and_report_keys() -> None:
    """Verify duplicate detection and required summary keys in report."""
    df = pd.DataFrame(
        {
            "cik": ["0001", "0001", "0002"],
            "accession_number": ["A1", "A1", "A2"],
            "ticker": ["ABC", "ABC", "XYZ"],
            "filing_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "assets": [100.0, 100.0, 200.0],
        }
    )

    anomalies, report = validate_and_detect(df)

    assert "status" in report
    assert "row_count" in report
    assert "anomaly_count" in report
    assert report["duplicate_count_cik_accession_number"] == 1
    assert report["row_count"] == 3
    assert report["anomaly_count"] == len(anomalies)


def test_anomaly_rows_generated_for_extreme_values() -> None:
    """Verify extreme numeric values are flagged as anomalies."""
    df = pd.DataFrame(
        {
            "cik": ["0001"] * 10,
            "accession_number": [f"A{i}" for i in range(10)],
            "ticker": ["ABC"] * 10,
            "filing_date": pd.to_datetime(["2024-01-01"] * 10),
            "assets": [10, 11, 10, 12, 11, 10, 12, 11, 10, 1000],
            "liabilities": [5, 5, 6, 5, 6, 5, 6, 5, 5, 800],
        }
    )

    anomalies, report = validate_and_detect(df)

    assert len(anomalies) >= 1
    assert report["anomaly_count"] >= 1
