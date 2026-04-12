"""Integration tests for drift monitor summary contract."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.monitoring import drift_monitor as dm


def _make_reference_df() -> pd.DataFrame:
    """Build a minimal reference dataset with required identity + numeric fields."""
    return pd.DataFrame(
        {
            "firm_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "date": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"]),
            "fiscal_year": [2023, 2023, 2023, 2023],
            "fiscal_period": ["Q1", "Q2", "Q3", "Q4"],
            "distress_label": [0, 0, 1, 0],
            "total_assets": [1000.0, 1200.0, 800.0, 900.0],
            "total_liabilities": [500.0, 700.0, 550.0, 600.0],
            "net_income": [80.0, 95.0, -20.0, 35.0],
            "gdp_growth": [2.1, 2.0, 1.8, 1.9],
        }
    )


def _make_current_df() -> pd.DataFrame:
    """Build a minimal current dataset with shifted numeric distributions."""
    return pd.DataFrame(
        {
            "firm_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "date": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]),
            "fiscal_year": [2024, 2024, 2024, 2024],
            "fiscal_period": ["Q1", "Q2", "Q3", "Q4"],
            "distress_label": [0, 1, 1, 1],
            "total_assets": [700.0, 760.0, 680.0, 720.0],
            "total_liabilities": [620.0, 640.0, 610.0, 630.0],
            "net_income": [-10.0, -15.0, -8.0, -12.0],
            "gdp_growth": [1.1, 1.0, 0.9, 1.0],
        }
    )


def test_run_drift_monitor_summary_schema(monkeypatch):
    """Run Evidently on sample data and assert the summary output schema."""
    reference_df = _make_reference_df()
    current_df = _make_current_df()

    upload_calls: list[str] = []
    retrain_calls: list[tuple[str, list[str]]] = []

    def fake_read_parquet(path: str, *args, **kwargs) -> pd.DataFrame:
        if path == dm.REFERENCE_PATH:
            return reference_df
        if path == dm.CURRENT_PATH:
            return current_df
        raise AssertionError(f"Unexpected parquet path: {path}")

    def fake_upload(_local_path, gcs_path: str) -> None:
        upload_calls.append(gcs_path)

    def fake_retrain_flag(reason: str, drifted_features: list[str]) -> None:
        retrain_calls.append((reason, drifted_features))

    monkeypatch.setattr(dm.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(dm, "_upload_to_gcs", fake_upload)
    monkeypatch.setattr(dm, "_write_retrain_flag", fake_retrain_flag)

    summary = dm.run_drift_monitor()

    expected_keys = {
        "date",
        "dataset_drift",
        "drift_share",
        "n_drifted_features",
        "drifted_features",
        "n_features_analyzed",
        "retrain_triggered",
    }
    assert expected_keys.issubset(summary.keys())

    datetime.strptime(summary["date"], "%Y-%m-%d")
    assert isinstance(summary["dataset_drift"], bool)
    assert isinstance(summary["drift_share"], float)
    assert 0.0 <= summary["drift_share"] <= 1.0
    assert isinstance(summary["n_drifted_features"], int)
    assert summary["n_drifted_features"] >= 0
    assert isinstance(summary["drifted_features"], list)
    assert len(summary["drifted_features"]) <= 10
    assert isinstance(summary["n_features_analyzed"], int)
    assert summary["n_features_analyzed"] > 0
    assert isinstance(summary["retrain_triggered"], bool)

    assert any(path.endswith("summary_latest.json") for path in upload_calls)
    assert any(path.endswith(".html") for path in upload_calls)

    if summary["retrain_triggered"]:
        assert len(retrain_calls) == 1
        assert retrain_calls[0][0] == "dataset_drift"
    else:
        assert retrain_calls == []
