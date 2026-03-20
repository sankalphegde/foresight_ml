"""Tests for SHAP sensitivity analysis (explain.py) and model-level bias fairness.

Covers:
  - SHAP value computation on a small synthetic model
  - Feature importance bar plot generation
  - Beeswarm plot generation
  - Top-20 summary table construction
  - Per-row top_features_json derivation
  - SHAP parquet serialization
  - Model-level fairness computation and bias alert detection
  - Threshold adjustment suggestions
  - Bias report Markdown generation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.models.explain import (
    build_top_features_table,
    compute_shap_values,
    derive_top_features_json,
    save_beeswarm_plot,
    save_feature_importance_bar_plot,
    save_shap_parquet,
)
from src.feature_engineering.pipelines.bias_analysis import (
    compute_model_fairness,
    generate_bias_report_md,
    suggest_threshold_adjustments,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_model_and_data():
    """Train a tiny XGBoost model on synthetic data for SHAP testing."""
    np.random.seed(42)
    n_train, n_test = 200, 50
    n_features = 10

    feature_names = [f"feat_{i}" for i in range(n_features)]

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features), columns=feature_names
    )
    # Simple rule: distress if feat_0 > 0.5 and feat_1 < -0.3
    y_train = ((X_train["feat_0"] > 0.5) & (X_train["feat_1"] < -0.3)).astype(int)

    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features), columns=feature_names
    )
    y_test = ((X_test["feat_0"] > 0.5) & (X_test["feat_1"] < -0.3)).astype(int)

    model = XGBClassifier(
        n_estimators=20,
        max_depth=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # Build a test_df that includes identifiers and the label
    test_df = X_test.copy()
    test_df["distress_label"] = y_test
    test_df["firm_id"] = [f"FIRM_{i:03d}" for i in range(n_test)]
    test_df["fiscal_year"] = 2022
    test_df["fiscal_period"] = "Q4"

    return model, X_test, test_df, feature_names


@pytest.fixture
def sample_slice_performance():
    """Mock slice performance table matching evaluate.py output format."""
    return pd.DataFrame([
        {"dimension": "company_size", "slice": "small", "sample_count": 100,
         "roc_auc": 0.72, "recall_at_5pct": 0.60, "precision_at_5pct": 0.40},
        {"dimension": "company_size", "slice": "mid", "sample_count": 200,
         "roc_auc": 0.85, "recall_at_5pct": 0.80, "precision_at_5pct": 0.55},
        {"dimension": "company_size", "slice": "large", "sample_count": 150,
         "roc_auc": 0.88, "recall_at_5pct": 0.82, "precision_at_5pct": 0.60},
        {"dimension": "company_size", "slice": "mega", "sample_count": 50,
         "roc_auc": 0.90, "recall_at_5pct": 0.85, "precision_at_5pct": 0.65},
        {"dimension": "sector_proxy", "slice": "tech_pharma", "sample_count": 120,
         "roc_auc": 0.86, "recall_at_5pct": 0.78, "precision_at_5pct": 0.50},
        {"dimension": "sector_proxy", "slice": "financial", "sample_count": 130,
         "roc_auc": 0.65, "recall_at_5pct": 0.45, "precision_at_5pct": 0.30},
        {"dimension": "sector_proxy", "slice": "manufacturing_retail", "sample_count": 150,
         "roc_auc": 0.84, "recall_at_5pct": 0.76, "precision_at_5pct": 0.48},
        {"dimension": "sector_proxy", "slice": "services", "sample_count": 100,
         "roc_auc": 0.82, "recall_at_5pct": 0.74, "precision_at_5pct": 0.45},
    ])


# ═══════════════════════════════════════════════════════════════════════════
# SHAP Computation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeShapValues:
    def test_shape_matches_input(self, synthetic_model_and_data):
        model, X_test, _, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        assert shap_vals.shape == X_test.shape

    def test_returns_numpy_array(self, synthetic_model_and_data):
        model, X_test, _, _ = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        assert isinstance(shap_vals, np.ndarray)

    def test_shap_values_are_finite(self, synthetic_model_and_data):
        model, X_test, _, _ = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        assert np.all(np.isfinite(shap_vals))


# ═══════════════════════════════════════════════════════════════════════════
# Plot Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPlots:
    def test_bar_plot_saved(self, synthetic_model_and_data, tmp_path):
        model, X_test, _, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        out = tmp_path / "bar.png"
        result = save_feature_importance_bar_plot(shap_vals, feature_names, out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_beeswarm_plot_saved(self, synthetic_model_and_data, tmp_path):
        model, X_test, _, _ = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        out = tmp_path / "beeswarm.png"
        result = save_beeswarm_plot(shap_vals, X_test, out)
        assert result.exists()
        assert result.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════
# Top Features Table Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTopFeaturesTable:
    def test_table_has_expected_columns(self, synthetic_model_and_data):
        model, X_test, _, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        table = build_top_features_table(shap_vals, feature_names, top_n=5)
        assert "feature" in table.columns
        assert "mean_abs_shap" in table.columns
        assert "direction" in table.columns

    def test_table_respects_top_n(self, synthetic_model_and_data):
        model, X_test, _, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        table = build_top_features_table(shap_vals, feature_names, top_n=3)
        assert len(table) == 3

    def test_table_sorted_by_importance(self, synthetic_model_and_data):
        model, X_test, _, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        table = build_top_features_table(shap_vals, feature_names, top_n=5)
        vals = table["mean_abs_shap"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_direction_values(self, synthetic_model_and_data):
        model, X_test, _, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        table = build_top_features_table(shap_vals, feature_names, top_n=5)
        valid_directions = {"increases_risk", "protective"}
        assert set(table["direction"].unique()).issubset(valid_directions)


# ═══════════════════════════════════════════════════════════════════════════
# Per-Row JSON Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDeriveTopFeaturesJson:
    def test_length_matches_samples(self, synthetic_model_and_data):
        model, X_test, _, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        result = derive_top_features_json(shap_vals, feature_names, top_k=3)
        assert len(result) == len(X_test)

    def test_json_is_valid(self, synthetic_model_and_data):
        model, X_test, _, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        result = derive_top_features_json(shap_vals, feature_names, top_k=3)
        for j in result:
            parsed = json.loads(j)
            assert isinstance(parsed, list)
            assert len(parsed) == 3

    def test_json_has_expected_keys(self, synthetic_model_and_data):
        model, X_test, _, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        result = derive_top_features_json(shap_vals, feature_names, top_k=2)
        first = json.loads(result[0])
        assert all(
            {"feature", "shap_value", "rank"} == set(entry.keys())
            for entry in first
        )

    def test_ranks_are_sequential(self, synthetic_model_and_data):
        model, X_test, _, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        result = derive_top_features_json(shap_vals, feature_names, top_k=3)
        first = json.loads(result[0])
        ranks = [entry["rank"] for entry in first]
        assert ranks == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════
# SHAP Parquet Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestShapParquet:
    def test_parquet_saved(self, synthetic_model_and_data, tmp_path):
        model, X_test, test_df, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        top_json = derive_top_features_json(shap_vals, feature_names, top_k=3)
        out = tmp_path / "shap.parquet"
        result = save_shap_parquet(shap_vals, feature_names, test_df, top_json, out)
        assert result.exists()

    def test_parquet_has_id_columns(self, synthetic_model_and_data, tmp_path):
        model, X_test, test_df, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        top_json = derive_top_features_json(shap_vals, feature_names, top_k=3)
        out = tmp_path / "shap.parquet"
        save_shap_parquet(shap_vals, feature_names, test_df, top_json, out)
        loaded = pd.read_parquet(out)
        assert "firm_id" in loaded.columns
        assert "top_features_json" in loaded.columns

    def test_parquet_row_count(self, synthetic_model_and_data, tmp_path):
        model, X_test, test_df, feature_names = synthetic_model_and_data
        shap_vals = compute_shap_values(model, X_test)
        top_json = derive_top_features_json(shap_vals, feature_names, top_k=3)
        out = tmp_path / "shap.parquet"
        save_shap_parquet(shap_vals, feature_names, test_df, top_json, out)
        loaded = pd.read_parquet(out)
        assert len(loaded) == len(test_df)


# ═══════════════════════════════════════════════════════════════════════════
# Model-Level Fairness Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeModelFairness:
    def test_returns_dataframe(self, sample_slice_performance):
        result = compute_model_fairness(sample_slice_performance)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, sample_slice_performance):
        result = compute_model_fairness(sample_slice_performance)
        expected = {"dimension", "slice", "metric", "slice_value",
                    "overall_value", "gap", "bias_alert", "sample_count"}
        assert expected.issubset(set(result.columns))

    def test_detects_bias_alert(self, sample_slice_performance):
        """The 'financial' sector has roc_auc=0.65 and recall=0.45,
        which should be flagged as bias alerts."""
        result = compute_model_fairness(sample_slice_performance)
        alerts = result[result["bias_alert"]]
        alerted_slices = set(alerts["slice"])
        assert "financial" in alerted_slices

    def test_no_alert_for_good_slices(self, sample_slice_performance):
        result = compute_model_fairness(sample_slice_performance)
        mega = result[(result["slice"] == "mega") & (result["metric"] == "roc_auc")]
        if not mega.empty:
            assert not mega.iloc[0]["bias_alert"]

    def test_custom_threshold(self, sample_slice_performance):
        """With a very large threshold, no alerts should fire."""
        result = compute_model_fairness(
            sample_slice_performance, alert_threshold=0.99
        )
        assert not result["bias_alert"].any()

    def test_empty_input(self):
        empty = pd.DataFrame(
            columns=["dimension", "slice", "sample_count", "roc_auc", "recall_at_5pct"]
        )
        result = compute_model_fairness(empty)
        assert result.empty


class TestSuggestThresholdAdjustments:
    def test_returns_suggestions_for_alerts(self, sample_slice_performance):
        fairness = compute_model_fairness(sample_slice_performance)
        suggestions = suggest_threshold_adjustments(fairness, base_threshold=0.5)
        # The 'financial' slice should get a suggestion
        if not suggestions.empty:
            assert "suggested_threshold" in suggestions.columns
            assert all(suggestions["suggested_threshold"] < 0.5)

    def test_no_suggestions_when_no_alerts(self, sample_slice_performance):
        fairness = compute_model_fairness(
            sample_slice_performance, alert_threshold=0.99
        )
        suggestions = suggest_threshold_adjustments(fairness)
        assert suggestions.empty

    def test_threshold_floor(self, sample_slice_performance):
        fairness = compute_model_fairness(sample_slice_performance)
        suggestions = suggest_threshold_adjustments(
            fairness, base_threshold=0.5, adjustment_step=0.05
        )
        if not suggestions.empty:
            assert all(suggestions["suggested_threshold"] >= 0.1)


# ═══════════════════════════════════════════════════════════════════════════
# Bias Report Markdown Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateBiasReportMd:
    def test_report_is_string(self, sample_slice_performance):
        feature_report = pd.DataFrame([
            {"dimension": "company_size", "slice": "small", "sample_count": 100}
        ])
        details = {"alerts": [], "drift_matrices": {}, "slices": {}}
        fairness = compute_model_fairness(sample_slice_performance)
        suggestions = suggest_threshold_adjustments(fairness)

        report = generate_bias_report_md(
            feature_report, details, fairness, suggestions
        )
        assert isinstance(report, str)
        assert "# Bias Report" in report

    def test_report_contains_sections(self, sample_slice_performance):
        feature_report = pd.DataFrame([
            {"dimension": "company_size", "slice": "small", "sample_count": 100}
        ])
        details = {"alerts": ["⚠ HIGH DRIFT: test alert"], "drift_matrices": {}, "slices": {}}
        fairness = compute_model_fairness(sample_slice_performance)
        suggestions = suggest_threshold_adjustments(fairness)

        report = generate_bias_report_md(
            feature_report, details, fairness, suggestions
        )
        assert "Feature-Level Drift" in report
        assert "Model-Level Fairness" in report
        assert "Mitigation" in report

    def test_report_saves_to_file(self, sample_slice_performance, tmp_path):
        feature_report = pd.DataFrame()
        details = {"alerts": []}
        fairness = compute_model_fairness(sample_slice_performance)
        suggestions = suggest_threshold_adjustments(fairness)

        out = tmp_path / "bias_report.md"
        generate_bias_report_md(
            feature_report, details, fairness, suggestions, output_path=str(out)
        )
        assert out.exists()
        content = out.read_text()
        assert "Bias Report" in content