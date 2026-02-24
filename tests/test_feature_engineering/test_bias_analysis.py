"""Unit Tests — Bias Analysis.

Tests for:
  - Slice creation
  - Per-slice statistics
  - PSI computation
  - Missing label handling
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.pipelines.bias_analysis import (
    analyze_slice_statistics,
    compute_js_divergence,
    compute_psi,
    create_slices,
    run_bias_analysis,
)

# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_engineered_df():
    """DataFrame with engineered features for bias testing."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "firm_id": [f"000000{i}" for i in range(10)] * 10,
        "fiscal_year": sorted([2014, 2015, 2016, 2017, 2018] * 20),
        "fiscal_period": (["Q1", "Q2", "Q3", "Q4"] * 25),
        "Assets": np.random.uniform(1e8, 1e11, n),
        "current_ratio": np.random.uniform(0.5, 3.0, n),
        "debt_to_equity": np.random.uniform(0.2, 5.0, n),
        "net_margin": np.random.uniform(-0.3, 0.4, n),
        "roa": np.random.uniform(-0.1, 0.2, n),
        "operating_margin": np.random.uniform(-0.2, 0.4, n),
        "cash_flow_to_debt": np.random.uniform(-0.1, 0.5, n),
        "altman_z_approx": np.random.uniform(0.5, 4.0, n),
        "cash_burn_rate": np.random.uniform(-0.5, 0.5, n),
        "rd_intensity": np.random.uniform(0, 0.3, n),
        "sga_intensity": np.random.uniform(0.05, 0.4, n),
        "leverage_x_margin": np.random.uniform(-1, 2, n),
        "company_size_bucket": np.random.choice(
            ["small", "mid", "large", "mega"], n
        ),
        "sector_proxy": np.random.choice(
            ["tech_pharma", "manufacturing_retail", "financial_capital_intensive", "services_other"], n
        ),
        "fed_funds": np.random.choice([0.5, 1.25, 2.5, 4.0], n),
    })
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PSI Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPSI:
    """Tests for Population Stability Index computation."""

    def test_identical_distributions(self):
        """Identical distributions should have PSI close to zero."""
        data = pd.Series(np.random.normal(0, 1, 1000))
        psi = compute_psi(data, data)
        assert psi < 0.01, "Identical distributions should have PSI ≈ 0"

    def test_different_distributions(self):
        """Shifted distributions should have PSI > 0.1."""
        ref = pd.Series(np.random.normal(0, 1, 1000))
        comp = pd.Series(np.random.normal(3, 1, 1000))
        psi = compute_psi(ref, comp)
        assert psi > 0.1, "Shifted distributions should have PSI > 0.1"

    def test_small_sample_returns_nan(self):
        """Samples with fewer than 10 elements should return NaN."""
        ref = pd.Series([1, 2, 3])
        comp = pd.Series([4, 5, 6])
        psi = compute_psi(ref, comp)
        assert np.isnan(psi)

    def test_constant_returns_zero(self):
        """Constant-valued distributions should produce PSI of zero."""
        ref = pd.Series([5.0] * 100)
        comp = pd.Series([5.0] * 100)
        psi = compute_psi(ref, comp)
        assert psi == 0.0


class TestJSDivergence:
    """Tests for Jensen-Shannon divergence computation."""

    def test_identical_distributions(self):
        """Identical distributions should have near-zero JS divergence."""
        data = pd.Series(np.random.normal(0, 1, 1000))
        js = compute_js_divergence(data, data)
        assert js < 0.05

    def test_different_distributions(self):
        """Shifted distributions should have JS divergence > 0.1."""
        ref = pd.Series(np.random.normal(0, 1, 1000))
        comp = pd.Series(np.random.normal(5, 1, 1000))
        js = compute_js_divergence(ref, comp)
        assert js > 0.1


# ═══════════════════════════════════════════════════════════════════════════
# Slice Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSliceCreation:
    """Tests for data slice creation across bias dimensions."""

    def test_creates_all_dimensions(self, sample_engineered_df):
        """All four slicing dimensions should be present."""
        slices = create_slices(sample_engineered_df)
        assert "company_size" in slices
        assert "sector_proxy" in slices
        assert "time_period" in slices
        assert "macro_regime" in slices

    def test_size_slices_cover_all_data(self, sample_engineered_df):
        """Size slices should cover every row with no gaps."""
        slices = create_slices(sample_engineered_df)
        total = sum(len(s) for s in slices["company_size"].values())
        assert total == len(sample_engineered_df)

    def test_time_split(self, sample_engineered_df):
        """Time split should partition data by fiscal_year threshold."""
        slices = create_slices(sample_engineered_df, time_split_year=2016)
        pre = slices["time_period"]["pre_2016"]
        post = slices["time_period"]["post_2016"]
        assert (pre["fiscal_year"] < 2016).all()
        assert (post["fiscal_year"] >= 2016).all()

    def test_macro_regime_split(self, sample_engineered_df):
        """Macro regime split should partition by fed_funds threshold."""
        slices = create_slices(sample_engineered_df, fed_funds_threshold=2.0)
        low = slices["macro_regime"]["low_rate"]
        high = slices["macro_regime"]["high_rate"]
        assert (low["fed_funds"] <= 2.0).all()
        assert (high["fed_funds"] > 2.0).all()


# ═══════════════════════════════════════════════════════════════════════════
# Statistics Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSliceStatistics:
    """Tests for per-slice summary statistics computation."""

    def test_statistics_computed(self, sample_engineered_df):
        """All expected statistics columns should be present."""
        stats = analyze_slice_statistics(
            sample_engineered_df, ["current_ratio", "net_margin"]
        )
        assert "sample_count" in stats
        assert "current_ratio_mean" in stats
        assert "current_ratio_std" in stats
        assert "net_margin_median" in stats
        assert "net_margin_missing_rate" in stats
        assert "net_margin_outlier_rate" in stats

    def test_sample_count_correct(self, sample_engineered_df):
        """Sample count should match the length of the input DataFrame."""
        stats = analyze_slice_statistics(sample_engineered_df, ["current_ratio"])
        assert stats["sample_count"] == len(sample_engineered_df)


# ═══════════════════════════════════════════════════════════════════════════
# Full Bias Analysis Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRunBiasAnalysis:
    """Tests for the full bias analysis pipeline."""

    def test_returns_report_and_details(self, sample_engineered_df):
        """Pipeline should return a DataFrame report and a details dict."""
        report, details = run_bias_analysis(sample_engineered_df)
        assert isinstance(report, pd.DataFrame)
        assert isinstance(details, dict)
        assert "slices" in details
        assert "drift_matrices" in details
        assert "alerts" in details

    def test_report_has_expected_columns(self, sample_engineered_df):
        """Bias report should contain dimension, slice, and sample_count columns."""
        report, _ = run_bias_analysis(sample_engineered_df)
        assert "dimension" in report.columns
        assert "slice" in report.columns
        assert "sample_count" in report.columns

    def test_no_crash_without_labels(self, sample_engineered_df):
        """Pipeline should work even without distress labels."""
        report, details = run_bias_analysis(sample_engineered_df)
        assert len(report) > 0
