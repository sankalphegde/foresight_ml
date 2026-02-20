"""Unit tests for src/data/preprocess.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# Adjust import path so tests can find src/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import preprocess_sec, preprocess_fred, build_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_sec_df() -> pd.DataFrame:
    """Minimal raw SEC DataFrame matching the raw layer schema."""
    return pd.DataFrame({
        "cik": ["1750", "1750", "1750", "1750", "1750"],
        "tag": ["Assets", "Assets", "Liabilities", "Assets", "Revenue"],
        "unit": ["USD", "USD", "USD", "USD", "USD"],
        "value": [1000.0, 1100.0, 500.0, 900.0, 200.0],
        "start_date": [None, None, None, None, "2020-01-01"],
        "end_date": ["2020-03-31", "2020-03-31", "2020-03-31", "2020-06-30", "2020-03-31"],
        "fiscal_year": [2020, 2020, 2020, 2020, 2020],
        "fiscal_period": ["Q1", "Q1", "Q1", "Q2", "Q1"],
        "form": ["10-Q", "10-Q", "10-Q", "10-Q", "10-Q"],
        "filed_date": ["2020-05-01", "2020-05-15", "2020-05-01", "2020-08-01", "2020-05-01"],
        "frame": [None, None, None, None, None],
        "quarter_key": ["2020_Q1", "2020_Q1", "2020_Q1", "2020_Q2", "2020_Q1"],
    })


@pytest.fixture
def raw_fred_df() -> pd.DataFrame:
    """Minimal raw FRED DataFrame."""
    return pd.DataFrame({
        "date": ["2020-03-31", "2020-06-30", "2020-03-31", "2020-09-30"],
        "value": [1.5, 0.25, 1.5, 0.10],
        "series_id": ["DFF", "DFF", "DFF", "DFF"],
    })


# ---------------------------------------------------------------------------
# SEC Tests
# ---------------------------------------------------------------------------

class TestPreprocessSec:

    def test_cik_zero_padded(self, raw_sec_df: pd.DataFrame):
        result = preprocess_sec(raw_sec_df)
        assert all(len(c) == 10 for c in result["cik"]), "CIK should be zero-padded to 10 digits"

    def test_drops_null_fiscal_year(self, raw_sec_df: pd.DataFrame):
        df = raw_sec_df.copy()
        df.loc[0, "fiscal_year"] = None
        result = preprocess_sec(df)
        assert result["fiscal_year"].isna().sum() == 0

    def test_filters_invalid_fiscal_period(self, raw_sec_df: pd.DataFrame):
        df = raw_sec_df.copy()
        df.loc[0, "fiscal_period"] = "Q5"
        result = preprocess_sec(df)
        assert "Q5" not in result["fiscal_period"].values

    def test_valid_fiscal_periods_kept(self, raw_sec_df: pd.DataFrame):
        result = preprocess_sec(raw_sec_df)
        valid = {"Q1", "Q2", "Q3", "Q4", "FY"}
        assert set(result["fiscal_period"].unique()).issubset(valid)

    def test_deduplication_keeps_latest(self, raw_sec_df: pd.DataFrame):
        """Row 0 and 1 are both (cik=1750, 2020, Q1, Assets). Row 1 has later filed_date."""
        result = preprocess_sec(raw_sec_df)
        assets_q1 = result[
            (result["tag"] == "Assets")
            & (result["fiscal_period"] == "Q1")
            & (result["fiscal_year"] == 2020)
        ]
        assert len(assets_q1) == 1, "Should deduplicate to one row"
        assert assets_q1.iloc[0]["value"] == 1100.0, "Should keep the row with latest filed_date"

    def test_quarter_key_format(self, raw_sec_df: pd.DataFrame):
        result = preprocess_sec(raw_sec_df)
        for _, row in result.iterrows():
            expected = f"{row['fiscal_year']}_{row['fiscal_period']}"
            assert row["quarter_key"] == expected

    def test_filters_fiscal_year_out_of_range(self, raw_sec_df: pd.DataFrame):
        df = raw_sec_df.copy()
        df.loc[0, "fiscal_year"] = 44012  # Excel date serial
        df.loc[1, "fiscal_year"] = 1985   # Too old
        result = preprocess_sec(df)
        assert result["fiscal_year"].min() >= 1990
        assert result["fiscal_year"].max() <= 2030

    def test_drops_quarter_end_date_if_mostly_null(self, raw_sec_df: pd.DataFrame):
        df = raw_sec_df.copy()
        df["quarter_end_date"] = None  # 100% null
        result = preprocess_sec(df)
        assert "quarter_end_date" not in result.columns

    def test_output_has_no_nulls_in_key_columns(self, raw_sec_df: pd.DataFrame):
        result = preprocess_sec(raw_sec_df)
        for col in ["cik", "fiscal_year", "fiscal_period", "tag", "quarter_key"]:
            assert result[col].isna().sum() == 0, f"{col} should have no nulls"


# ---------------------------------------------------------------------------
# FRED Tests
# ---------------------------------------------------------------------------

class TestPreprocessFred:

    def test_deduplication(self, raw_fred_df: pd.DataFrame):
        """Row 0 and 2 are duplicates (DFF, 2020-03-31)."""
        result = preprocess_fred(raw_fred_df)
        dff_q1 = result[(result["series_id"] == "DFF") & (result["date"].dt.month == 3)]
        assert len(dff_q1) == 1, "Should deduplicate by (series_id, date)"

    def test_date_snapped_to_quarter_end(self, raw_fred_df: pd.DataFrame):
        result = preprocess_fred(raw_fred_df)
        for d in result["date"]:
            assert d == d + pd.offsets.QuarterEnd(0), f"{d} is not quarter-end"

    def test_sorted_ascending(self, raw_fred_df: pd.DataFrame):
        result = preprocess_fred(raw_fred_df)
        dates = result["date"].tolist()
        assert dates == sorted(dates)

    def test_drops_null_dates(self, raw_fred_df: pd.DataFrame):
        df = raw_fred_df.copy()
        df.loc[0, "date"] = None
        result = preprocess_fred(df)
        assert result["date"].isna().sum() == 0

    def test_value_is_numeric(self, raw_fred_df: pd.DataFrame):
        result = preprocess_fred(raw_fred_df)
        assert result["value"].dtype in ["float64", "float32", "int64"]

    def test_series_id_preserved(self, raw_fred_df: pd.DataFrame):
        result = preprocess_fred(raw_fred_df)
        assert "series_id" in result.columns
        assert result["series_id"].nunique() == 1


# ---------------------------------------------------------------------------
# Report Tests
# ---------------------------------------------------------------------------

class TestBuildReport:

    def test_report_structure(self, raw_sec_df: pd.DataFrame, raw_fred_df: pd.DataFrame):
        sec = preprocess_sec(raw_sec_df)
        fred = preprocess_fred(raw_fred_df)
        report = build_report(sec, fred)

        assert "sec" in report
        assert "fred" in report
        assert "unique_ciks" in report
        assert "unique_fred_series" in report
        assert report["sec"]["row_count"] == len(sec)
        assert report["fred"]["row_count"] == len(fred)

    def test_report_is_json_serializable(self, raw_sec_df: pd.DataFrame, raw_fred_df: pd.DataFrame):
        sec = preprocess_sec(raw_sec_df)
        fred = preprocess_fred(raw_fred_df)
        report = build_report(sec, fred)
        # Should not raise
        json.dumps(report, default=str)