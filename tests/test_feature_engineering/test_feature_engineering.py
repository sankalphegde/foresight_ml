"""Unit Tests — Feature Engineering & Data Cleaning.

Tests for:
  - NaN imputation/dropping logic
  - Financial ratio computation
  - Growth rate computation
  - Rolling statistics
  - Outlier clipping
  - Division-by-zero guarding
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.pipelines.data_cleaning import (
    clean_data,
    drop_uninformative_columns,
    impute_financial_columns,
    impute_macro_columns,
)
from src.feature_engineering.pipelines.feature_engineering import (
    clip_outliers,
    compute_financial_ratios,
    compute_growth_rates,
    compute_rolling_stats,
    engineer_features,
    safe_divide,
)

# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Create a small DataFrame mimicking the raw data schema."""
    np.random.seed(42)
    n = 20

    df = pd.DataFrame({
        "firm_id": ["0000001"] * 10 + ["0000002"] * 10,
        "ticker": ["TEST1"] * 10 + ["TEST2"] * 10,
        "fiscal_year": [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021, 2022, 2022] * 2,
        "fiscal_period": ["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4", "Q1", "Q2"] * 2,
        "filed_date": pd.date_range("2020-03-01", periods=10, freq="QS").tolist() * 2,
        "Assets": np.random.uniform(1e9, 1e11, n),
        "AssetsCurrent": np.random.uniform(5e8, 5e10, n),
        "CashAndCashEquivalentsAtCarryingValue": np.random.uniform(1e8, 1e10, n),
        "InventoryNet": np.random.uniform(1e7, 5e9, n),
        "AccountsReceivableNetCurrent": np.random.uniform(1e8, 5e9, n),
        "PropertyPlantAndEquipmentNet": np.random.uniform(1e8, 1e10, n),
        "Goodwill": np.random.uniform(0, 5e9, n),
        "IntangibleAssetsNetExcludingGoodwill": np.random.uniform(0, 3e9, n),
        "Liabilities": np.random.uniform(5e8, 5e10, n),
        "LiabilitiesCurrent": np.random.uniform(1e8, 1e10, n),
        "AccountsPayableCurrent": np.random.uniform(1e7, 3e9, n),
        "LongTermDebt": np.random.uniform(0, 2e10, n),
        "LongTermDebtCurrent": np.random.uniform(0, 2e9, n),
        "StockholdersEquity": np.random.uniform(1e8, 5e10, n),
        "RetainedEarningsAccumulatedDeficit": np.random.uniform(-5e9, 2e10, n),
        "AdditionalPaidInCapital": np.random.uniform(0, 1e10, n),
        "CommonStockValue": np.random.uniform(0, 5e9, n),
        "Revenues": np.random.uniform(1e9, 5e10, n),
        "CostOfGoodsAndServicesSold": np.random.uniform(5e8, 3e10, n),
        "GrossProfit": np.random.uniform(1e8, 2e10, n),
        "OperatingIncomeLoss": np.random.uniform(-5e9, 1e10, n),
        "NetIncomeLoss": np.random.uniform(-5e9, 8e9, n),
        "ResearchAndDevelopmentExpense": np.random.uniform(0, 5e9, n),
        "SellingGeneralAndAdministrativeExpense": np.random.uniform(1e8, 5e9, n),
        "InterestExpense": np.random.uniform(0, 1e9, n),
        "IncomeTaxExpenseBenefit": np.random.uniform(-1e9, 3e9, n),
        "EarningsPerShareBasic": [np.nan] * n,
        "EarningsPerShareDiluted": [np.nan] * n,
        "NetCashProvidedByUsedInOperatingActivities": np.random.uniform(-2e9, 1e10, n),
        "NetCashProvidedByUsedInInvestingActivities": np.random.uniform(-1e10, 2e9, n),
        "NetCashProvidedByUsedInFinancingActivities": np.random.uniform(-5e9, 5e9, n),
        "DepreciationDepletionAndAmortization": np.random.uniform(0, 3e9, n),
        "fed_funds": [1.25] * 5 + [np.nan] * 5 + [2.5] * 5 + [np.nan] * 5,
        "unemployment": [3.8] * 5 + [np.nan] * 5 + [4.2] * 5 + [np.nan] * 5,
        "inflation": [258.0] * 5 + [np.nan] * 5 + [270.0] * 5 + [np.nan] * 5,
        "quality_check_flag": ["Valid"] * n,
    })
    return df


@pytest.fixture
def sample_clean_df(sample_raw_df):
    """Cleaned version of the sample data."""
    return clean_data(sample_raw_df.copy())


# ═══════════════════════════════════════════════════════════════════════════
# Data Cleaning Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDropColumns:
    """Tests for dropping uninformative columns."""

    def test_drops_eps_columns(self, sample_raw_df):
        """EPS columns (100% null) should be dropped."""
        result = drop_uninformative_columns(sample_raw_df.copy())
        assert "EarningsPerShareBasic" not in result.columns
        assert "EarningsPerShareDiluted" not in result.columns

    def test_drops_quality_check_flag(self, sample_raw_df):
        """Quality check flag (zero-variance) should be dropped."""
        result = drop_uninformative_columns(sample_raw_df.copy())
        assert "quality_check_flag" not in result.columns

    def test_preserves_other_columns(self, sample_raw_df):
        """Non-dropped columns should remain intact."""
        result = drop_uninformative_columns(sample_raw_df.copy())
        assert "Assets" in result.columns
        assert "Revenues" in result.columns
        assert "firm_id" in result.columns or "cik" in result.columns


class TestImputeMacroColumns:
    """Tests for macroeconomic column imputation."""

    def test_no_nulls_after_imputation(self, sample_raw_df):
        """All macro columns should have zero nulls after imputation."""
        result = impute_macro_columns(sample_raw_df.copy())
        assert result["fed_funds"].isnull().sum() == 0
        assert result["unemployment"].isnull().sum() == 0
        assert result["inflation"].isnull().sum() == 0

    def test_ffill_logic(self, sample_raw_df):
        """After ffill, the nulls should take the prior row's value."""
        df = sample_raw_df.copy()
        result = impute_macro_columns(df)
        # All values should be filled — no NaN left
        assert result["fed_funds"].notna().all()


class TestImputeFinancialColumns:
    """Tests for financial statement column imputation."""

    def test_handles_already_clean_data(self, sample_raw_df):
        """Financial columns have 0% null — should pass through unchanged."""
        result = impute_financial_columns(sample_raw_df.copy())
        assert result["Assets"].isnull().sum() == 0

    def test_fills_injected_nulls(self, sample_raw_df):
        """Inject NaN and verify imputation."""
        df = sample_raw_df.copy()
        df.loc[3, "Assets"] = np.nan
        result = impute_financial_columns(df)
        assert result["Assets"].isnull().sum() == 0


class TestCleanData:
    """Tests for the full data cleaning pipeline."""

    def test_full_clean_pipeline(self, sample_raw_df):
        """Full pipeline should drop columns, impute, and preserve rows."""
        result = clean_data(sample_raw_df.copy())
        # Dropped columns gone
        assert "EarningsPerShareBasic" not in result.columns
        # No nulls in macro columns
        assert result["fed_funds"].isnull().sum() == 0
        # Row count preserved
        assert len(result) == len(sample_raw_df)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Engineering Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSafeDivide:
    """Tests for safe_divide helper guarding against zero denominators."""

    def test_normal_division(self):
        """Standard division should produce correct results."""
        num = pd.Series([10.0, 20.0, 30.0])
        denom = pd.Series([2.0, 4.0, 5.0])
        result = safe_divide(num, denom)
        np.testing.assert_array_almost_equal(result, [5.0, 5.0, 6.0])

    def test_zero_denominator_returns_nan(self):
        """Zero denominator should produce NaN, not raise."""
        num = pd.Series([10.0, 20.0])
        denom = pd.Series([0.0, 5.0])
        result = safe_divide(num, denom)
        assert np.isnan(result[0])
        assert result[1] == 4.0

    def test_both_zero(self):
        """Both zero should produce NaN."""
        num = pd.Series([0.0])
        denom = pd.Series([0.0])
        result = safe_divide(num, denom)
        assert np.isnan(result[0])


class TestFinancialRatios:
    """Tests for financial ratio computation."""

    def test_ratios_created(self, sample_clean_df):
        """All 13 expected ratio columns should exist after computation."""
        result = compute_financial_ratios(sample_clean_df.copy())
        expected_ratios = [
            "current_ratio", "quick_ratio", "cash_ratio",
            "debt_to_equity", "debt_to_assets", "interest_coverage",
            "gross_margin", "operating_margin", "net_margin",
            "roa", "roe", "asset_turnover", "cash_flow_to_debt",
        ]
        for ratio in expected_ratios:
            assert ratio in result.columns, f"Missing ratio: {ratio}"

    def test_current_ratio_formula(self, sample_clean_df):
        """Current ratio should equal AssetsCurrent / LiabilitiesCurrent."""
        result = compute_financial_ratios(sample_clean_df.copy())
        expected = sample_clean_df["AssetsCurrent"] / sample_clean_df["LiabilitiesCurrent"]
        # Only check where denominator is non-zero
        mask = sample_clean_df["LiabilitiesCurrent"] != 0
        np.testing.assert_array_almost_equal(
            result.loc[mask, "current_ratio"].values,
            expected.loc[mask].values,
        )

    def test_no_inf_values(self, sample_clean_df):
        """Financial ratios should not contain infinite values."""
        result = compute_financial_ratios(sample_clean_df.copy())
        for col in ["current_ratio", "debt_to_equity", "gross_margin"]:
            assert not np.isinf(result[col].dropna()).any(), f"Inf in {col}"


class TestGrowthRates:
    """Tests for year-over-year growth rate computation."""

    def test_growth_columns_created(self, sample_clean_df):
        """Growth rate columns should be created with expected names."""
        result = compute_growth_rates(sample_clean_df.copy(), lag=4)
        assert "revenue_growth_yoy" in result.columns
        assert "assets_growth_yoy" in result.columns

    def test_first_periods_are_nan(self, sample_clean_df):
        """First 4 quarters per company should be NaN (no prior year data)."""
        result = compute_growth_rates(sample_clean_df.copy(), lag=4)
        company_1 = result[result["firm_id"] == "0000001"].sort_values(
            ["fiscal_year", "fiscal_period"]
        )
        # First 4 rows should be NaN for growth
        assert company_1["revenue_growth_yoy"].iloc[:4].isnull().all()


class TestRollingStats:
    """Tests for rolling statistics computation."""

    def test_rolling_columns_created(self, sample_clean_df):
        """Rolling mean and std columns should be created."""
        df = compute_financial_ratios(sample_clean_df.copy())
        df = compute_growth_rates(df, lag=4)
        result = compute_rolling_stats(df, windows=[4])
        assert "current_ratio_rolling_4q_mean" in result.columns
        assert "current_ratio_rolling_4q_std" in result.columns


class TestClipOutliers:
    """Tests for outlier clipping functionality."""

    def test_clipping_works(self):
        """Extreme values should be clipped within ±n_std."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        result = clip_outliers(df.copy(), ["x"], n_std=2.0)
        assert result["x"].max() < 100


class TestEngineerFeatures:
    """Tests for the full feature engineering pipeline."""

    def test_full_pipeline(self, sample_clean_df):
        """Pipeline should add columns and preserve rows."""
        result = engineer_features(sample_clean_df.copy())
        # Should have more columns than input
        assert result.shape[1] > sample_clean_df.shape[1]
        # Row count preserved
        assert result.shape[0] == sample_clean_df.shape[0]
        # Key features exist
        assert "altman_z_approx" in result.columns
        assert "company_size_bucket" in result.columns
        assert "sector_proxy" in result.columns

    def test_no_inf_in_output(self, sample_clean_df):
        """Output should contain no infinite values."""
        result = engineer_features(sample_clean_df.copy())
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Infinite values in output"
