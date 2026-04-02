"""Read / write / validate manifest.json files.

Separated from ``manifest_schema.py`` so that:
- The schema module is pure validation (no I/O side effects).
- This module handles filesystem and GCS operations independently.
- Both modules are independently testable.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from src.models.manifest_schema import ManifestSchema

log = logging.getLogger(__name__)


def write_manifest(manifest: ManifestSchema, local_path: Path) -> None:
    """Serialize and write a validated manifest to a local JSON file.

    Creates parent directories if they do not exist.

    Args:
        manifest: Validated ManifestSchema instance.
        local_path: Local filesystem path to write to.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(manifest.model_dump_json(indent=2))
    log.info("Wrote manifest.json to %s", local_path)


def upload_manifest_to_gcs(local_path: Path, gcs_dir: str) -> str:
    """Upload a local manifest.json to GCS alongside its scores.parquet.

    Args:
        local_path: Path to the local manifest.json.
        gcs_dir: GCS directory prefix
            (e.g. ``gs://bucket/inference/scores_v1.0/``).

    Returns:
        Full GCS URI of the uploaded manifest.
    """
    gcs_path = f"{gcs_dir.rstrip('/')}/manifest.json"
    try:
        subprocess.run(
            ["gsutil", "cp", str(local_path), gcs_path],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("Uploaded manifest.json -> %s", gcs_path)
    except FileNotFoundError:
        log.warning("gsutil not found; skipping upload of %s", gcs_path)
    except subprocess.CalledProcessError as e:
        log.warning("gsutil upload failed: %s", e.stderr.strip())
    return gcs_path


def read_manifest(path: Path) -> ManifestSchema:
    """Read and validate a manifest.json from a local file.

    Args:
        path: Path to the JSON file.

    Returns:
        Validated ManifestSchema instance.

    Raises:
        pydantic.ValidationError: If the JSON doesn't match ManifestSchema.
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    raw = json.loads(path.read_text())
    return ManifestSchema(**raw)


def validate_manifest_dict(data: dict) -> ManifestSchema:
    """Validate a raw dict against the ManifestSchema.

    Useful for testing or validating manifests fetched from GCS
    without first writing to disk.

    Args:
        data: Dictionary to validate.

    Returns:
        Validated ManifestSchema instance.

    Raises:
        pydantic.ValidationError: If the dict doesn't match ManifestSchema.
    """
    return ManifestSchema(**data)
