"""Tests for inference input/output schema validation."""

from __future__ import annotations

import pytest
import pandas as pd
from pydantic import ValidationError

from src.models.inference_schema import (
    IDENTITY_COLUMNS,
    LABEL_COLUMN,
    InferenceOutputRow,
    validate_inference_input,
    validate_inference_output,
)


# ---------- Helpers ----------


def _make_input_df(**overrides) -> pd.DataFrame:
    """Create a minimal valid input DataFrame for testing."""
    base = {
        "firm_id": ["0000001750"],
        "fiscal_year": [2025],
        "fiscal_period": ["Q4"],
        "total_assets": [1_000_000.0],
        "total_liabilities": [500_000.0],
        "net_income": [50_000.0],
        "BBB_spread": [1.5],
        "CPI": [310.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_output_df(**overrides) -> pd.DataFrame:
    """Create a minimal valid output DataFrame for testing."""
    base = {
        "firm_id": ["0000001750"],
        "fiscal_year": [2025],
        "fiscal_period": ["Q4"],
        "distress_probability": [0.12],
        "model_version": ["v1"],
        "mlflow_run_id": ["abc123"],
        "trained_at": ["2026-03-15T10:00:00+00:00"],
        "scored_at": ["2026-03-30T06:00:00+00:00"],
        "model_roc_auc": [0.977],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------- Constants Tests ----------


class TestConstants:

    def test_identity_columns_has_three(self):
        assert len(IDENTITY_COLUMNS) == 3

    def test_identity_columns_content(self):
        assert "firm_id" in IDENTITY_COLUMNS
        assert "fiscal_year" in IDENTITY_COLUMNS
        assert "fiscal_period" in IDENTITY_COLUMNS

    def test_label_column_value(self):
        assert LABEL_COLUMN == "distress_label"


# ---------- InferenceOutputRow Tests ----------


class TestInferenceOutputRow:

    def test_valid_output_row(self):
        row = InferenceOutputRow(
            firm_id="0000001750",
            fiscal_year=2025,
            fiscal_period="Q4",
            distress_probability=0.12,
            model_version="v1",
            mlflow_run_id="abc123",
            trained_at="2026-03-15T10:00:00+00:00",
            scored_at="2026-03-30T06:00:00+00:00",
            model_roc_auc=0.977,
        )
        assert row.distress_probability == 0.12

    def test_confidence_intervals_optional(self):
        """Confidence intervals should be Optional per user requirement."""
        row = InferenceOutputRow(
            firm_id="0000001750",
            fiscal_year=2025,
            fiscal_period="Q4",
            distress_probability=0.5,
            model_version="v1",
            mlflow_run_id="abc123",
            trained_at="2026-03-15T10:00:00+00:00",
            scored_at="2026-03-30T06:00:00+00:00",
            model_roc_auc=0.977,
        )
        assert row.confidence_interval_lower is None
        assert row.confidence_interval_upper is None

    def test_confidence_intervals_accepted_when_present(self):
        row = InferenceOutputRow(
            firm_id="0000001750",
            fiscal_year=2025,
            fiscal_period="Q4",
            distress_probability=0.5,
            confidence_interval_lower=0.45,
            confidence_interval_upper=0.55,
            model_version="v1",
            mlflow_run_id="abc123",
            trained_at="2026-03-15T10:00:00+00:00",
            scored_at="2026-03-30T06:00:00+00:00",
            model_roc_auc=0.977,
        )
        assert row.confidence_interval_lower == 0.45

    def test_probability_above_1_raises(self):
        with pytest.raises(ValidationError):
            InferenceOutputRow(
                firm_id="0000001750",
                fiscal_year=2025,
                fiscal_period="Q4",
                distress_probability=1.5,
                model_version="v1",
                mlflow_run_id="abc123",
                trained_at="2026-03-15T10:00:00+00:00",
                scored_at="2026-03-30T06:00:00+00:00",
                model_roc_auc=0.977,
            )

    def test_model_version_without_v_raises(self):
        with pytest.raises(ValidationError):
            InferenceOutputRow(
                firm_id="0000001750",
                fiscal_year=2025,
                fiscal_period="Q4",
                distress_probability=0.12,
                model_version="1",
                mlflow_run_id="abc123",
                trained_at="2026-03-15T10:00:00+00:00",
                scored_at="2026-03-30T06:00:00+00:00",
                model_roc_auc=0.977,
            )


# ---------- validate_inference_input Tests ----------


class TestValidateInferenceInput:

    def test_valid_df_no_errors(self):
        df = _make_input_df()
        errors = validate_inference_input(df)
        assert errors == []

    def test_missing_identity_column_detected(self):
        df = _make_input_df()
        df = df.drop(columns=["firm_id"])
        errors = validate_inference_input(df)
        assert any("Missing identity column" in e for e in errors)

    def test_label_column_presence_detected(self):
        df = _make_input_df(distress_label=[0])
        errors = validate_inference_input(df)
        assert any("Label column" in e for e in errors)

    def test_non_numeric_feature_detected(self):
        df = _make_input_df(total_assets=["not_a_number"])
        errors = validate_inference_input(df)
        assert any("Non-numeric" in e for e in errors)

    def test_all_null_column_detected(self):
        df = _make_input_df(total_assets=[None])
        errors = validate_inference_input(df)
        assert any("entirely null" in e for e in errors)

    def test_multiple_errors_returned(self):
        """Should return ALL errors, not just the first."""
        df = _make_input_df(distress_label=[0], total_assets=["bad"])
        errors = validate_inference_input(df)
        assert len(errors) >= 2


# ---------- validate_inference_output Tests ----------


class TestValidateInferenceOutput:

    def test_valid_output_no_errors(self):
        df = _make_output_df()
        errors = validate_inference_output(df)
        assert errors == []

    def test_missing_output_column_detected(self):
        df = _make_output_df()
        df = df.drop(columns=["distress_probability"])
        errors = validate_inference_output(df)
        assert any("Missing output columns" in e for e in errors)

    def test_probability_out_of_range_detected(self):
        df = _make_output_df(distress_probability=[1.5])
        errors = validate_inference_output(df)
        assert any("outside [0, 1]" in e for e in errors)

    def test_null_versioning_column_detected(self):
        df = _make_output_df(model_version=[None])
        errors = validate_inference_output(df)
        assert any("null values" in e for e in errors)
