"""Tests for manifest.json schema validation and I/O."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.models.manifest_io import read_manifest, validate_manifest_dict, write_manifest
from src.models.manifest_schema import ManifestSchema

# ---------- Fixtures ----------


@pytest.fixture()
def valid_manifest_data() -> dict:
    """Minimal valid manifest data matching ManifestSchema."""
    return {
        "schema_version": "1.0",
        "model_name": "foresight_xgboost",
        "model_version": "v1",
        "mlflow_run_id": "abc123def456",
        "trained_at": "2026-03-15T10:00:00+00:00",
        "scored_at": "2026-03-30T06:00:00+00:00",
        "roc_auc": 0.9769,
        "prediction_horizon": 1,
        "features_used": ["total_assets", "net_income"],
        "row_count": 48291,
        "gcs_scores_path": "gs://bucket/inference/v1/scores.parquet",
        "inference_duration_seconds": 42.7,
    }


@pytest.fixture()
def valid_manifest(valid_manifest_data: dict) -> ManifestSchema:
    """A valid ManifestSchema instance."""
    return ManifestSchema(**valid_manifest_data)


# ---------- Schema Validation Tests ----------


class TestManifestSchemaValidation:
    def test_valid_manifest_passes(self, valid_manifest_data):
        m = ManifestSchema(**valid_manifest_data)
        assert m.model_name == "foresight_xgboost"

    def test_missing_required_field_raises(self, valid_manifest_data):
        del valid_manifest_data["model_version"]
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_roc_auc_above_1_raises(self, valid_manifest_data):
        valid_manifest_data["roc_auc"] = 1.5
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_roc_auc_negative_raises(self, valid_manifest_data):
        valid_manifest_data["roc_auc"] = -0.1
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_row_count_zero_raises(self, valid_manifest_data):
        valid_manifest_data["row_count"] = 0
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_row_count_negative_raises(self, valid_manifest_data):
        valid_manifest_data["row_count"] = -1
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_gcs_path_without_gs_prefix_raises(self, valid_manifest_data):
        valid_manifest_data["gcs_scores_path"] = "s3://bucket/scores.parquet"
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_gcs_path_without_parquet_suffix_raises(self, valid_manifest_data):
        valid_manifest_data["gcs_scores_path"] = "gs://bucket/scores.csv"
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_model_version_without_v_prefix_raises(self, valid_manifest_data):
        valid_manifest_data["model_version"] = "1"
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_empty_features_list_raises(self, valid_manifest_data):
        valid_manifest_data["features_used"] = []
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_invalid_schema_version_format_raises(self, valid_manifest_data):
        valid_manifest_data["schema_version"] = "v1"
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_prediction_horizon_zero_raises(self, valid_manifest_data):
        valid_manifest_data["prediction_horizon"] = 0
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_negative_inference_duration_raises(self, valid_manifest_data):
        valid_manifest_data["inference_duration_seconds"] = -1.0
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_empty_model_name_raises(self, valid_manifest_data):
        valid_manifest_data["model_name"] = ""
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)

    def test_empty_mlflow_run_id_raises(self, valid_manifest_data):
        valid_manifest_data["mlflow_run_id"] = ""
        with pytest.raises(ValidationError):
            ManifestSchema(**valid_manifest_data)


# ---------- Serialization Tests ----------


class TestManifestSerialization:
    def test_round_trip_json(self, valid_manifest):
        json_str = valid_manifest.model_dump_json(indent=2)
        restored = ManifestSchema.model_validate_json(json_str)
        assert restored == valid_manifest

    def test_all_12_fields_present_in_json(self, valid_manifest):
        data = json.loads(valid_manifest.model_dump_json())
        expected_fields = {
            "schema_version",
            "model_name",
            "model_version",
            "mlflow_run_id",
            "trained_at",
            "scored_at",
            "roc_auc",
            "prediction_horizon",
            "features_used",
            "row_count",
            "gcs_scores_path",
            "inference_duration_seconds",
        }
        assert set(data.keys()) == expected_fields

    def test_timestamps_serialize_as_iso8601(self, valid_manifest):
        from datetime import datetime

        data = json.loads(valid_manifest.model_dump_json())
        datetime.fromisoformat(data["trained_at"])
        datetime.fromisoformat(data["scored_at"])

    def test_json_schema_generation(self):
        schema = ManifestSchema.model_json_schema()
        assert "properties" in schema
        assert "model_name" in schema["properties"]
        assert "roc_auc" in schema["properties"]


# ---------- I/O Tests ----------


class TestManifestIO:
    def test_write_and_read_roundtrip(self, tmp_path, valid_manifest):
        path = tmp_path / "manifest.json"
        write_manifest(valid_manifest, path)
        assert path.exists()
        restored = read_manifest(path)
        assert restored == valid_manifest

    def test_write_creates_parent_dirs(self, tmp_path, valid_manifest):
        path = tmp_path / "deep" / "nested" / "manifest.json"
        write_manifest(valid_manifest, path)
        assert path.exists()

    def test_read_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            read_manifest(path)

    def test_read_missing_fields_raises(self, tmp_path):
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"model_name": "test"}))
        with pytest.raises(ValidationError):
            read_manifest(path)

    def test_read_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path / "does_not_exist.json")

    def test_validate_manifest_dict(self, valid_manifest_data):
        m = validate_manifest_dict(valid_manifest_data)
        assert m.row_count == 48291

    def test_validate_manifest_dict_invalid_raises(self):
        with pytest.raises(ValidationError):
            validate_manifest_dict({"model_name": "test"})
