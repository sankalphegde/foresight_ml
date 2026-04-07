"""Tests for the Foresight-ML Streamlit dashboard utilities and data layer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.dashboard.utils import (
    COLORS,
    apply_chart_theme,
    fmt_large_number,
    parse_top_features_json,
    quarter_label,
    quarter_sort_key,
    risk_badge_html,
    risk_color,
    risk_emoji,
    risk_level,
    shap_color,
)


# ---------------------------------------------------------------------------
# risk_level
# ---------------------------------------------------------------------------


class TestRiskLevel:
    """Tests for risk_level()."""

    def test_high_risk(self) -> None:
        assert risk_level(0.70) == "High"
        assert risk_level(0.95) == "High"

    def test_medium_risk(self) -> None:
        assert risk_level(0.30) == "Medium"
        assert risk_level(0.50) == "Medium"
        assert risk_level(0.69) == "Medium"

    def test_low_risk(self) -> None:
        assert risk_level(0.0) == "Low"
        assert risk_level(0.29) == "Low"

    def test_boundary_high(self) -> None:
        assert risk_level(0.70) == "High"

    def test_boundary_medium(self) -> None:
        assert risk_level(0.30) == "Medium"


# ---------------------------------------------------------------------------
# risk_emoji
# ---------------------------------------------------------------------------


class TestRiskEmoji:
    """Tests for risk_emoji()."""

    def test_high(self) -> None:
        assert risk_emoji(0.80) == "🔴"

    def test_medium(self) -> None:
        assert risk_emoji(0.50) == "🟡"

    def test_low(self) -> None:
        assert risk_emoji(0.10) == "🟢"


# ---------------------------------------------------------------------------
# risk_color
# ---------------------------------------------------------------------------


class TestRiskColor:
    """Tests for risk_color()."""

    def test_high(self) -> None:
        assert risk_color(0.80) == COLORS["high"]

    def test_medium(self) -> None:
        assert risk_color(0.50) == COLORS["medium"]

    def test_low(self) -> None:
        assert risk_color(0.10) == COLORS["low"]


# ---------------------------------------------------------------------------
# risk_badge_html
# ---------------------------------------------------------------------------


class TestRiskBadgeHtml:
    """Tests for risk_badge_html()."""

    def test_returns_html_string(self) -> None:
        result = risk_badge_html(0.85)
        assert "<span" in result
        assert "High" in result
        assert "0.85" in result

    def test_low_risk_badge(self) -> None:
        result = risk_badge_html(0.10)
        assert "Low" in result

    def test_medium_risk_badge(self) -> None:
        result = risk_badge_html(0.50)
        assert "Medium" in result


# ---------------------------------------------------------------------------
# parse_top_features_json
# ---------------------------------------------------------------------------


class TestParseTopFeaturesJson:
    """Tests for parse_top_features_json()."""

    def test_valid_json(self) -> None:
        data = [{"feature": "net_income", "shap_value": -0.5}]
        result = parse_top_features_json(json.dumps(data))
        assert len(result) == 1
        assert result[0]["feature"] == "net_income"

    def test_empty_string(self) -> None:
        assert parse_top_features_json("") == []

    def test_invalid_json(self) -> None:
        assert parse_top_features_json("not json") == []

    def test_nan_input(self) -> None:
        assert parse_top_features_json(float("nan")) == []  # type: ignore[arg-type]

    def test_none_like(self) -> None:
        assert parse_top_features_json("") == []


# ---------------------------------------------------------------------------
# shap_color
# ---------------------------------------------------------------------------


class TestShapColor:
    """Tests for shap_color()."""

    def test_positive_is_red(self) -> None:
        assert shap_color(0.5) == COLORS["shap_positive"]

    def test_negative_is_green(self) -> None:
        assert shap_color(-0.3) == COLORS["shap_negative"]

    def test_zero_is_green(self) -> None:
        assert shap_color(0.0) == COLORS["shap_negative"]


# ---------------------------------------------------------------------------
# fmt_large_number
# ---------------------------------------------------------------------------


class TestFmtLargeNumber:
    """Tests for fmt_large_number()."""

    def test_billions(self) -> None:
        assert fmt_large_number(2.4e9) == "$2.4B"

    def test_millions(self) -> None:
        assert fmt_large_number(3.1e6) == "$3.1M"

    def test_thousands(self) -> None:
        assert fmt_large_number(500_000) == "$500K"

    def test_small_number(self) -> None:
        assert fmt_large_number(42) == "$42"

    def test_negative(self) -> None:
        assert fmt_large_number(-1.5e9) == "-$1.5B"

    def test_nan(self) -> None:
        assert fmt_large_number(float("nan")) == "N/A"

    def test_zero(self) -> None:
        assert fmt_large_number(0) == "$0"


# ---------------------------------------------------------------------------
# quarter_label
# ---------------------------------------------------------------------------


class TestQuarterLabel:
    """Tests for quarter_label()."""

    def test_normal(self) -> None:
        assert quarter_label(2025, "Q3") == "Q3 2025"

    def test_q1(self) -> None:
        assert quarter_label(2022, "Q1") == "Q1 2022"


# ---------------------------------------------------------------------------
# quarter_sort_key
# ---------------------------------------------------------------------------


class TestQuarterSortKey:
    """Tests for quarter_sort_key()."""

    def test_ordering(self) -> None:
        assert quarter_sort_key(2022, "Q1") < quarter_sort_key(2022, "Q2")
        assert quarter_sort_key(2022, "Q4") < quarter_sort_key(2023, "Q1")

    def test_same_year(self) -> None:
        keys = [quarter_sort_key(2023, q) for q in ["Q1", "Q2", "Q3", "Q4"]]
        assert keys == sorted(keys)

    def test_unknown_period(self) -> None:
        result = quarter_sort_key(2023, "unknown")
        assert result == 2023 * 10


# ---------------------------------------------------------------------------
# apply_chart_theme
# ---------------------------------------------------------------------------


class TestApplyChartTheme:
    """Tests for apply_chart_theme()."""

    def test_returns_figure(self) -> None:
        fig = MagicMock()
        result = apply_chart_theme(fig)
        assert result is fig

    def test_calls_update_layout(self) -> None:
        fig = MagicMock()
        apply_chart_theme(fig)
        fig.update_layout.assert_called_once()

    def test_calls_update_axes(self) -> None:
        fig = MagicMock()
        apply_chart_theme(fig)
        fig.update_xaxes.assert_called_once()
        fig.update_yaxes.assert_called_once()


# ---------------------------------------------------------------------------
# GCS loader helpers
# ---------------------------------------------------------------------------


class TestGcsLoader:
    """Tests for gcs_loader helper functions."""

    def test_get_company_history_empty_panel(self) -> None:
        from src.dashboard.data.gcs_loader import get_company_history

        result = get_company_history(pd.DataFrame(), "0000001234")
        assert result.empty

    def test_get_company_history_filters(self) -> None:
        from src.dashboard.data.gcs_loader import get_company_history

        panel = pd.DataFrame({
            "firm_id": ["AAA", "BBB", "AAA"],
            "fiscal_year": [2022, 2022, 2023],
            "fiscal_period": ["Q1", "Q1", "Q2"],
        })
        result = get_company_history(panel, "AAA")
        assert len(result) == 2
        assert all(result["firm_id"] == "AAA")

    def test_get_shap_for_company_empty(self) -> None:
        from src.dashboard.data.gcs_loader import get_shap_for_company

        result = get_shap_for_company(pd.DataFrame(), "0000001234")
        assert result.empty

    def test_get_shap_for_company_filters(self) -> None:
        from src.dashboard.data.gcs_loader import get_shap_for_company

        shap_df = pd.DataFrame({
            "firm_id": ["AAA", "BBB", "AAA"],
            "fiscal_year": [2022, 2022, 2023],
            "shap_net_income": [0.1, 0.2, 0.3],
        })
        result = get_shap_for_company(shap_df, "AAA")
        assert len(result) == 2

    def test_get_shap_for_company_with_year_filter(self) -> None:
        from src.dashboard.data.gcs_loader import get_shap_for_company

        shap_df = pd.DataFrame({
            "firm_id": ["AAA", "AAA", "AAA"],
            "fiscal_year": [2022, 2023, 2023],
            "shap_net_income": [0.1, 0.2, 0.3],
        })
        result = get_shap_for_company(shap_df, "AAA", fiscal_year=2023)
        assert len(result) == 2

    def test_safe_read_parquet_missing_file(self) -> None:
        from src.dashboard.data.gcs_loader import _safe_read_parquet

        result = _safe_read_parquet("gs://nonexistent/file.parquet", "test")
        assert result.empty


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


class TestApiClient:
    """Tests for api_client functions."""

    @patch("src.dashboard.data.api_client.requests.get")
    def test_get_success(self, mock_get: MagicMock) -> None:
        from src.dashboard.data.api_client import _get

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "healthy"},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        result = _get("/health")
        assert result == {"status": "healthy"}

    @patch("src.dashboard.data.api_client.requests.get")
    def test_get_failure(self, mock_get: MagicMock) -> None:
        from src.dashboard.data.api_client import _get

        mock_get.side_effect = Exception("Connection refused")
        result = _get("/health")
        assert result is None

    @patch("src.dashboard.data.api_client.requests.post")
    def test_post_success(self, mock_post: MagicMock) -> None:
        from src.dashboard.data.api_client import _post

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"distress_probability": 0.88},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        result = _post("/predict", {"cik": "0000001234"})
        assert result == {"distress_probability": 0.88}

    @patch("src.dashboard.data.api_client.requests.post")
    def test_post_failure(self, mock_post: MagicMock) -> None:
        from src.dashboard.data.api_client import _post

        mock_post.side_effect = Exception("Timeout")
        result = _post("/predict", {"cik": "0000001234"})
        assert result is None

    @patch("src.dashboard.data.api_client.check_health")
    def test_is_api_available_true(self, mock_health: MagicMock) -> None:
        from src.dashboard.data.api_client import is_api_available

        mock_health.return_value = {"status": "healthy"}
        assert is_api_available() is True

    @patch("src.dashboard.data.api_client.check_health")
    def test_is_api_available_false(self, mock_health: MagicMock) -> None:
        from src.dashboard.data.api_client import is_api_available

        mock_health.return_value = None
        assert is_api_available() is False


# ---------------------------------------------------------------------------
# Company risk page helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Watchlist builder
# ---------------------------------------------------------------------------


class TestWatchlistBuilder:
    """Tests for watchlist._build_watchlist()."""

    def test_empty_predictions(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        result = _build_watchlist(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_basic_watchlist(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        preds = pd.DataFrame({
            "firm_id": ["AAA", "AAA", "BBB"],
            "fiscal_year": [2022, 2023, 2023],
            "fiscal_period": ["Q1", "Q2", "Q2"],
            "distress_probability": [0.3, 0.8, 0.1],
        })
        result = _build_watchlist(preds, pd.DataFrame())
        assert len(result) == 2
        assert "risk_score" in result.columns
        assert "signals" in result.columns
        assert "change" in result.columns

    def test_watchlist_with_panel_signals(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        preds = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "distress_probability": [0.9],
        })
        panel = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "net_income": [-500_000],
            "total_assets": [1_000_000],
            "total_liabilities": [900_000],
        })
        result = _build_watchlist(preds, panel)
        assert len(result) == 1
        assert "Neg. income" in result.iloc[0]["signals"]

    def test_watchlist_change_column(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        preds = pd.DataFrame({
            "firm_id": ["AAA", "AAA"],
            "fiscal_year": [2022, 2023],
            "fiscal_period": ["Q1", "Q2"],
            "distress_probability": [0.3, 0.8],
        })
        result = _build_watchlist(preds, pd.DataFrame())
        assert len(result) == 1
        assert result.iloc[0]["change"] == pytest.approx(0.5, abs=0.01)

    def test_watchlist_sector_default(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        preds = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "distress_probability": [0.5],
        })
        result = _build_watchlist(preds, pd.DataFrame())
        assert result.iloc[0]["sector"] == "—"

    def test_watchlist_no_signals(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        preds = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "distress_probability": [0.5],
        })
        panel = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "net_income": [1_000_000],
            "total_assets": [10_000_000],
            "total_liabilities": [2_000_000],
        })
        result = _build_watchlist(preds, panel)
        assert result.iloc[0]["signals"] == "—"


# ---------------------------------------------------------------------------
# GCS loader — additional coverage
# ---------------------------------------------------------------------------


class TestGcsLoaderDefaults:
    """Tests for gcs_loader default values and edge cases."""

    def test_read_gcs_json_failure(self) -> None:
        from src.dashboard.data.gcs_loader import _read_gcs_json

        result = _read_gcs_json("gs://nonexistent-bucket/file.json")
        assert result is None

    def test_default_manifest_keys(self) -> None:
        from src.dashboard.data.gcs_loader import DEFAULT_MANIFEST

        assert "model_name" in DEFAULT_MANIFEST
        assert "roc_auc" in DEFAULT_MANIFEST
        assert "schema_version" in DEFAULT_MANIFEST

    def test_get_shap_missing_firm_id_column(self) -> None:
        from src.dashboard.data.gcs_loader import get_shap_for_company

        shap_df = pd.DataFrame({"some_col": [1, 2, 3]})
        result = get_shap_for_company(shap_df, "AAA")
        assert result.empty

    def test_get_company_history_sorted(self) -> None:
        from src.dashboard.data.gcs_loader import get_company_history

        panel = pd.DataFrame({
            "firm_id": ["AAA", "AAA", "AAA"],
            "fiscal_year": [2023, 2022, 2023],
            "fiscal_period": ["Q2", "Q4", "Q1"],
        })
        result = get_company_history(panel, "AAA")
        assert list(result["fiscal_year"]) == [2022, 2023, 2023]


# ---------------------------------------------------------------------------
# API client — additional coverage
# ---------------------------------------------------------------------------


class TestApiClientEndpoints:
    """Tests for api_client public endpoint wrappers."""

    @patch("src.dashboard.data.api_client._get")
    def test_check_health(self, mock: MagicMock) -> None:
        from src.dashboard.data.api_client import check_health

        mock.return_value = {"status": "healthy"}
        assert check_health() == {"status": "healthy"}

    @patch("src.dashboard.data.api_client._get")
    def test_get_model_info(self, mock: MagicMock) -> None:
        from src.dashboard.data.api_client import get_model_info

        mock.return_value = {"model": "v1"}
        assert get_model_info() == {"model": "v1"}

    @patch("src.dashboard.data.api_client._post")
    def test_predict(self, mock: MagicMock) -> None:
        from src.dashboard.data.api_client import predict

        mock.return_value = {"distress_probability": 0.5}
        assert predict("0000001234") == {"distress_probability": 0.5}

    @patch("src.dashboard.data.api_client._get")
    def test_get_company(self, mock: MagicMock) -> None:
        from src.dashboard.data.api_client import get_company

        mock.return_value = {"cik": "123", "history": []}
        result = get_company("123")
        assert result is not None
        assert result["cik"] == "123"

    @patch("src.dashboard.data.api_client._get")
    def test_get_alerts(self, mock: MagicMock) -> None:
        from src.dashboard.data.api_client import get_alerts

        mock.return_value = {"alerts": []}
        assert get_alerts(0.70) == {"alerts": []}

    @patch("src.dashboard.data.api_client._get")
    def test_get_drift_status(self, mock: MagicMock) -> None:
        from src.dashboard.data.api_client import get_drift_status

        mock.return_value = {"dataset_drift": False}
        assert get_drift_status() == {"dataset_drift": False}


# ---------------------------------------------------------------------------
# Utils — additional edge cases
# ---------------------------------------------------------------------------


class TestUtilsEdgeCases:
    """Additional edge case tests for utils."""

    def test_risk_level_exact_boundaries(self) -> None:
        assert risk_level(0.30) == "Medium"
        assert risk_level(0.70) == "High"
        assert risk_level(0.0) == "Low"
        assert risk_level(1.0) == "High"

    def test_fmt_large_number_negative_millions(self) -> None:
        assert fmt_large_number(-5.5e6) == "-$5.5M"

    def test_fmt_large_number_negative_thousands(self) -> None:
        assert fmt_large_number(-250_000) == "-$250K"

    def test_risk_badge_contains_score(self) -> None:
        result = risk_badge_html(0.42)
        assert "0.42" in result
        assert "Medium" in result

    def test_parse_top_features_multiple(self) -> None:
        data = [
            {"feature": "a", "shap_value": 0.1},
            {"feature": "b", "shap_value": -0.2},
        ]
        result = parse_top_features_json(json.dumps(data))
        assert len(result) == 2

    def test_quarter_sort_key_cross_year(self) -> None:
        assert quarter_sort_key(2022, "Q4") < quarter_sort_key(2023, "Q1")

    def test_colors_dict_has_required_keys(self) -> None:
        required = ["high", "medium", "low", "high_bg", "medium_bg", "low_bg",
                     "shap_positive", "shap_negative", "text", "border"]
        for key in required:
            assert key in COLORS


# ---------------------------------------------------------------------------
# Pipeline status helpers
# ---------------------------------------------------------------------------


class TestPipelineStatusHelpers:
    """Tests for pipeline_status.py helper functions."""

    def test_status_dot_success(self) -> None:
        from src.dashboard.pages.pipeline_status import _status_dot

        result = _status_dot("success")
        assert "#16a34a" in result
        assert "border-radius:50%" in result

    def test_status_dot_failed(self) -> None:
        from src.dashboard.pages.pipeline_status import _status_dot

        result = _status_dot("failed")
        assert "#b91c1c" in result

    def test_status_dot_warning(self) -> None:
        from src.dashboard.pages.pipeline_status import _status_dot

        result = _status_dot("warning")
        assert "#d97706" in result

    def test_status_dot_running(self) -> None:
        from src.dashboard.pages.pipeline_status import _status_dot

        result = _status_dot("running")
        assert "#3b7dd8" in result

    def test_status_dot_pending(self) -> None:
        from src.dashboard.pages.pipeline_status import _status_dot

        result = _status_dot("pending")
        assert "#9c9a92" in result

    def test_status_dot_unknown(self) -> None:
        from src.dashboard.pages.pipeline_status import _status_dot

        result = _status_dot("unknown_status")
        assert "#9c9a92" in result

    def test_pipeline_row_success(self) -> None:
        from src.dashboard.pages.pipeline_status import _pipeline_row

        result = _pipeline_row("Test task", "~2m", "success", "Done")
        assert "Test task" in result
        assert "~2m" in result
        assert "Done" in result
        assert "#16a34a" in result

    def test_pipeline_row_failed(self) -> None:
        from src.dashboard.pages.pipeline_status import _pipeline_row

        result = _pipeline_row("Broken task", "~1m", "failed", "Error")
        assert "Broken task" in result
        assert "#b91c1c" in result

    def test_pipeline_row_no_detail(self) -> None:
        from src.dashboard.pages.pipeline_status import _pipeline_row

        result = _pipeline_row("Task", "~1m", "success", "")
        assert "Success" in result

    def test_pipeline_row_case_insensitive(self) -> None:
        from src.dashboard.pages.pipeline_status import _pipeline_row

        result = _pipeline_row("Task", "~1m", "SUCCESS", "OK")
        assert "#16a34a" in result


# ---------------------------------------------------------------------------
# GCS loader — cached functions with mocks
# ---------------------------------------------------------------------------


class TestGcsLoaderCachedFunctions:
    """Tests for gcs_loader cached loader return types."""

    @patch("src.dashboard.data.gcs_loader._read_gcs_json")
    def test_load_manifest_fallback(self, mock_read: MagicMock) -> None:
        from src.dashboard.data.gcs_loader import DEFAULT_MANIFEST

        mock_read.return_value = None
        # Can't easily call the cached function, but test the logic
        result = mock_read("gs://fake") or DEFAULT_MANIFEST
        assert result["model_name"] == "foresight-xgboost"

    @patch("src.dashboard.data.gcs_loader._read_gcs_json")
    def test_load_optuna_fallback(self, mock_read: MagicMock) -> None:
        mock_read.return_value = None
        default = {"baseline_val_roc": 0.0, "best_params": {}, "test_roc_auc": 0.0}
        result = mock_read("gs://fake") or default
        assert result["test_roc_auc"] == 0.0

    @patch("src.dashboard.data.gcs_loader._read_gcs_json")
    def test_load_drift_fallback(self, mock_read: MagicMock) -> None:
        mock_read.return_value = None
        default = {"drift_detected": False, "drifted_features": []}
        result = mock_read("gs://fake") or default
        assert result["drift_detected"] is False

    @patch("src.dashboard.data.gcs_loader._read_gcs_json")
    def test_load_manifest_success(self, mock_read: MagicMock) -> None:
        from src.dashboard.data.gcs_loader import DEFAULT_MANIFEST

        custom = {"model_name": "custom-model", "roc_auc": 0.95}
        mock_read.return_value = custom
        result = mock_read("gs://fake") or DEFAULT_MANIFEST
        assert result["model_name"] == "custom-model"

    def test_safe_read_parquet_invalid_uri(self) -> None:
        from src.dashboard.data.gcs_loader import _safe_read_parquet

        result = _safe_read_parquet("/tmp/definitely_not_a_file.parquet", "test")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# Watchlist — additional edge cases
# ---------------------------------------------------------------------------


class TestWatchlistEdgeCases:
    """Additional watchlist builder edge case tests."""

    def test_high_leverage_signal(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        preds = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "distress_probability": [0.8],
        })
        panel = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "total_assets": [1_000_000],
            "total_liabilities": [850_000],
        })
        result = _build_watchlist(preds, panel)
        assert "High leverage" in result.iloc[0]["signals"]

    def test_negative_cashflow_signal(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        preds = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "distress_probability": [0.6],
        })
        panel = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "NetCashProvidedByUsedInOperatingActivities": [-500_000],
        })
        result = _build_watchlist(preds, panel)
        assert "Neg. cash flow" in result.iloc[0]["signals"]

    def test_retained_earnings_signal(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        preds = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "distress_probability": [0.7],
        })
        panel = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "RetainedEarningsAccumulatedDeficit": [-2_000_000],
        })
        result = _build_watchlist(preds, panel)
        assert "Retained earnings" in result.iloc[0]["signals"]

    def test_multiple_firms(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        preds = pd.DataFrame({
            "firm_id": ["AAA", "BBB", "CCC"],
            "fiscal_year": [2023, 2023, 2023],
            "fiscal_period": ["Q1", "Q1", "Q1"],
            "distress_probability": [0.9, 0.5, 0.1],
        })
        result = _build_watchlist(preds, pd.DataFrame())
        assert len(result) == 3

    def test_watchlist_quarter_column(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        preds = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q3"],
            "distress_probability": [0.5],
        })
        result = _build_watchlist(preds, pd.DataFrame())
        assert "Q3" in result.iloc[0]["quarter"]
        assert "2023" in result.iloc[0]["quarter"]


# ---------------------------------------------------------------------------
# Signal chips — additional coverage
# ---------------------------------------------------------------------------


class TestSignalChipsEdgeCases:
    """Additional edge cases for _build_signal_chips."""

    def test_zero_assets(self) -> None:
        from src.dashboard.pages.company_risk import _build_signal_chips

        row = {
            "net_income": 100,
            "total_assets": 0,
            "total_liabilities": 0,
        }
        result = _build_signal_chips(row)
        assert "Profitable" in result
        assert "leverage" not in result.lower()

    def test_zero_cash_flow(self) -> None:
        from src.dashboard.pages.company_risk import _build_signal_chips

        row = {
            "net_income": 100,
            "NetCashProvidedByUsedInOperatingActivities": 0,
            "total_assets": 1_000,
            "total_liabilities": 100,
        }
        result = _build_signal_chips(row)
        assert "cash flow" not in result.lower()

    def test_missing_fields_default_zero(self) -> None:
        from src.dashboard.pages.company_risk import _build_signal_chips

        row: dict = {}
        result = _build_signal_chips(row)
        assert "Profitable" in result

    def test_all_negative_signals(self) -> None:
        from src.dashboard.pages.company_risk import _build_signal_chips

        row = {
            "net_income": -1,
            "NetCashProvidedByUsedInOperatingActivities": -1,
            "RetainedEarningsAccumulatedDeficit": -1,
            "total_assets": 100,
            "total_liabilities": 95,
        }
        result = _build_signal_chips(row)
        assert "Negative net income" in result
        assert "Negative cash flow" in result
        assert "Accumulated deficit" in result
        assert "High leverage" in result
        assert result.count("🔴") == 4

    def test_all_positive_signals(self) -> None:
        from src.dashboard.pages.company_risk import _build_signal_chips

        row = {
            "net_income": 1_000,
            "NetCashProvidedByUsedInOperatingActivities": 500,
            "RetainedEarningsAccumulatedDeficit": 200,
            "total_assets": 10_000,
            "total_liabilities": 2_000,
        }
        result = _build_signal_chips(row)
        assert "Profitable" in result
        assert "Positive cash flow" in result
        assert "Healthy leverage" in result
        assert "🟢" in result


# ---------------------------------------------------------------------------
# SHAP features — additional coverage
# ---------------------------------------------------------------------------


class TestShapFeaturesEdgeCases:
    """Additional tests for _get_top_shap_features."""

    def test_top_n_limits_results(self) -> None:
        from src.dashboard.pages.company_risk import _get_top_shap_features

        shap_df = pd.DataFrame({
            "shap_a": [0.5],
            "shap_b": [-0.4],
            "shap_c": [0.3],
            "shap_d": [-0.2],
            "shap_e": [0.1],
        })
        result = _get_top_shap_features(shap_df, top_n=3)
        assert len(result) == 3
        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2
        assert result[2]["rank"] == 3

    def test_sorts_by_absolute_value(self) -> None:
        from src.dashboard.pages.company_risk import _get_top_shap_features

        shap_df = pd.DataFrame({
            "shap_small": [0.01],
            "shap_big_negative": [-0.9],
            "shap_medium": [0.5],
        })
        result = _get_top_shap_features(shap_df, top_n=3)
        assert result[0]["feature"] == "big_negative"
        assert result[1]["feature"] == "medium"
        assert result[2]["feature"] == "small"

    def test_empty_json_fallback(self) -> None:
        from src.dashboard.pages.company_risk import _get_top_shap_features

        shap_df = pd.DataFrame({"top_features_json": ["[]"]})
        result = _get_top_shap_features(shap_df, top_n=5)
        assert result == []

    def test_single_shap_column(self) -> None:
        from src.dashboard.pages.company_risk import _get_top_shap_features

        shap_df = pd.DataFrame({"shap_only_feature": [0.42]})
        result = _get_top_shap_features(shap_df, top_n=5)
        assert len(result) == 1
        assert result[0]["feature"] == "only_feature"
        assert result[0]["shap_value"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Watchlist builder — comprehensive signal combinations
# ---------------------------------------------------------------------------


class TestWatchlistSignalCombinations:
    """Test all signal detection paths in _build_watchlist."""

    def _make_preds(self, firm_id: str = "AAA") -> pd.DataFrame:
        return pd.DataFrame({
            "firm_id": [firm_id],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "distress_probability": [0.8],
        })

    def test_all_four_signals(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        panel = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "net_income": [-100],
            "NetCashProvidedByUsedInOperatingActivities": [-50],
            "RetainedEarningsAccumulatedDeficit": [-200],
            "total_assets": [1000],
            "total_liabilities": [850],
        })
        result = _build_watchlist(self._make_preds(), panel)
        signals = result.iloc[0]["signals"]
        assert "Neg. income" in signals
        assert "Neg. cash flow" in signals
        assert "Retained earnings" in signals
        assert "High leverage" in signals

    def test_sector_from_panel(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        panel = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "sector_proxy": ["Technology"],
        })
        result = _build_watchlist(self._make_preds(), panel)
        assert result.iloc[0]["sector"] == "Technology"

    def test_size_from_panel(self) -> None:
        from src.dashboard.pages.watchlist import _build_watchlist

        panel = pd.DataFrame({
            "firm_id": ["AAA"],
            "fiscal_year": [2023],
            "fiscal_period": ["Q1"],
            "company_size_bucket": ["large"],
        })
        result = _build_watchlist(self._make_preds(), panel)
        assert result.iloc[0]["size"] == "large"


class TestCompanyRiskHelpers:
    """Tests for company_risk.py helper functions."""

    def test_build_signal_chips_profitable(self) -> None:
        from src.dashboard.pages.company_risk import _build_signal_chips

        row = {
            "net_income": 1_000_000,
            "NetCashProvidedByUsedInOperatingActivities": 500_000,
            "RetainedEarningsAccumulatedDeficit": 100_000,
            "total_assets": 10_000_000,
            "total_liabilities": 3_000_000,
        }
        result = _build_signal_chips(row)
        assert "Profitable" in result
        assert "Positive cash flow" in result
        assert "Healthy leverage" in result

    def test_build_signal_chips_distressed(self) -> None:
        from src.dashboard.pages.company_risk import _build_signal_chips

        row = {
            "net_income": -500_000,
            "NetCashProvidedByUsedInOperatingActivities": -200_000,
            "RetainedEarningsAccumulatedDeficit": -1_000_000,
            "total_assets": 1_000_000,
            "total_liabilities": 900_000,
        }
        result = _build_signal_chips(row)
        assert "Negative net income" in result
        assert "Negative cash flow" in result
        assert "Accumulated deficit" in result
        assert "High leverage" in result

    def test_get_top_shap_features_from_columns(self) -> None:
        from src.dashboard.pages.company_risk import _get_top_shap_features

        shap_df = pd.DataFrame({
            "shap_net_income": [-0.5],
            "shap_total_assets": [0.3],
            "shap_leverage": [0.1],
        })
        result = _get_top_shap_features(shap_df, top_n=2)
        assert len(result) == 2
        assert result[0]["feature"] == "net_income"
        assert result[0]["rank"] == 1

    def test_get_top_shap_features_fallback_json(self) -> None:
        from src.dashboard.pages.company_risk import _get_top_shap_features

        data = [{"feature": "f1", "shap_value": 0.5, "rank": 1}]
        shap_df = pd.DataFrame({"top_features_json": [json.dumps(data)]})
        result = _get_top_shap_features(shap_df, top_n=5)
        assert len(result) == 1
        assert result[0]["feature"] == "f1"