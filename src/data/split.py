"""Time-based train/validation/test data splitting utilities.

Splits financial distress data by fiscal_year with stratification,
fits a scaler/imputer pipeline on training data only, and serializes
everything to local disk (and optionally GCS).
"""

from __future__ import annotations

import json
import logging
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.settings import Settings, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COL = "distress_label"
STRAT_COLS = ("company_size_bucket", "sector_proxy")

# Columns that should never be scaled (identifiers, target, categoricals)
NON_NUMERIC_COLS = {
    "firm_id",
    "fiscal_year",
    "fiscal_period",
    "filed_date",
    TARGET_COL,
    "company_size_bucket",
    "sector_proxy",
    "_strat_key",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_features(
    source: str | Path | None = None,
    cfg: Settings = settings,
) -> pd.DataFrame:
    """Load feature data from BigQuery, CSV, or parquet.

    Args:
        source: Path to a local CSV/parquet file, or None to read from BigQuery.
        cfg: Settings instance for BigQuery table name.

    Returns:
        DataFrame with all features.
    """
    if source is not None:
        source = Path(source)
        if source.suffix == ".csv":
            log.info("Loading features from CSV: %s", source)
            return pd.read_csv(source)
        log.info("Loading features from parquet: %s", source)
        return pd.read_parquet(source)

    # BigQuery fallback
    log.info("Loading features from BigQuery: %s", cfg.bigquery_features_table)
    from google.cloud import bigquery  # noqa: E402

    client = bigquery.Client(project=cfg.project_id)
    query = f"SELECT * FROM `{cfg.bigquery_features_table}`"  # noqa: S608
    return client.query(query).to_dataframe()


# ---------------------------------------------------------------------------
# Stratification key
# ---------------------------------------------------------------------------


def make_stratification_key(
    df: pd.DataFrame,
    min_count: int = 10,
) -> pd.DataFrame:
    """Create a composite stratification column, merging rare strata.

    Builds a temporary ``_strat_key`` column from
    ``company_size_bucket × sector_proxy``.  Any combination that appears
    fewer than ``min_count`` times gets merged to ``"other"``.  The original
    columns are **never** modified.

    Args:
        df: Input DataFrame (must have STRAT_COLS).
        min_count: Minimum count threshold below which a stratum is merged.

    Returns:
        DataFrame with ``_strat_key`` column appended.
    """
    df = df.copy()
    raw_key = df[STRAT_COLS[0]].astype(str) + "__" + df[STRAT_COLS[1]].astype(str)
    counts = raw_key.value_counts()
    rare_keys = set(counts[counts < min_count].index)
    df["_strat_key"] = raw_key.where(~raw_key.isin(rare_keys), other="other")
    log.info(
        "Stratification: %d unique keys (%d merged into 'other')",
        df["_strat_key"].nunique(),
        len(rare_keys),
    )
    return df


# ---------------------------------------------------------------------------
# Time-based splitting
# ---------------------------------------------------------------------------


def time_based_split(
    df: pd.DataFrame,
    train_years: tuple[int, int] = settings.train_years,
    val_years: tuple[int, int] = settings.val_years,
    test_years: tuple[int, int] = settings.test_years,
    exclude_years: tuple[int, ...] = settings.exclude_years,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by fiscal_year into train / validation / test.

    Args:
        df: Full feature DataFrame.
        train_years: (start, end) inclusive range for training.
        val_years: (start, end) inclusive range for validation.
        test_years: (start, end) inclusive range for test.
        exclude_years: Years to drop entirely (e.g. 2009).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = df[~df["fiscal_year"].isin(exclude_years)].copy()
    log.info("Excluded years %s: %d rows remain", exclude_years, len(df))

    train = df[df["fiscal_year"].between(*train_years)].copy()
    val = df[df["fiscal_year"].between(*val_years)].copy()
    test = df[df["fiscal_year"].between(*test_years)].copy()

    log.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_no_temporal_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Assert that no temporal leakage exists across splits.

    Raises:
        AssertionError: If max(train.fiscal_year) >= min(val.fiscal_year)
            or max(val.fiscal_year) >= min(test.fiscal_year).
    """
    max_train = train["fiscal_year"].max()
    min_val = val["fiscal_year"].min()
    max_val = val["fiscal_year"].max()
    min_test = test["fiscal_year"].min()

    assert (
        max_train < min_val
    ), f"Temporal leakage: max train year {max_train} >= min val year {min_val}"
    assert (
        max_val < min_test
    ), f"Temporal leakage: max val year {max_val} >= min test year {min_test}"
    log.info("No temporal leakage: train<=%d < val<=%d < test<=%d", max_train, max_val, min_test)


def validate_stratification(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Verify all strata present in every split (best-effort).

    Logs a warning if not all strata are represented, rather than
    failing hard, since small sample sets may not cover all combos.
    """
    all_keys = set(train["_strat_key"].unique())
    for name, split in [("val", val), ("test", test)]:
        missing = all_keys - set(split["_strat_key"].unique())
        if missing:
            log.warning("Strata %s missing from %s split", missing, name)
        else:
            log.info("All %d strata present in %s split", len(all_keys), name)


# ---------------------------------------------------------------------------
# Scaler / imputer pipeline
# ---------------------------------------------------------------------------


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric feature columns (excluding identifiers/target)."""
    return [c for c in df.select_dtypes(include=np.number).columns if c not in NON_NUMERIC_COLS]


def fit_scaler(
    train_df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> tuple[Pipeline, list[str]]:
    """Fit a StandardScaler + SimpleImputer pipeline on training data only.

    Args:
        train_df: Training split DataFrame.
        numeric_cols: Columns to scale. If None, auto-detected.

    Returns:
        Tuple of (fitted Pipeline, list of column names).
    """
    if numeric_cols is None:
        numeric_cols = get_numeric_columns(train_df)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline.fit(train_df[numeric_cols])
    log.info("Fitted scaler pipeline on %d training columns", len(numeric_cols))
    return pipeline, numeric_cols


def apply_scaler(
    df: pd.DataFrame,
    pipeline: Pipeline,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """Apply a fitted scaler pipeline to a DataFrame.

    Args:
        df: DataFrame to transform.
        pipeline: Fitted sklearn Pipeline.
        numeric_cols: Columns the pipeline was fitted on.

    Returns:
        DataFrame with numeric columns replaced by scaled values.
    """
    df = df.copy()
    df[numeric_cols] = pipeline.transform(df[numeric_cols])
    return df


# ---------------------------------------------------------------------------
# Serialization / GCS upload
# ---------------------------------------------------------------------------


def _upload_to_gcs(local_path: Path, bucket: str, gcs_path: str) -> None:
    """Upload a local file to GCS using gsutil (same pattern as preprocess.py)."""
    dest = f"gs://{bucket}/{gcs_path}"
    try:
        subprocess.run(
            ["gsutil", "cp", str(local_path), dest],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("Uploaded -> %s", dest)
    except FileNotFoundError:
        log.warning("gsutil not found; skipping upload of %s", dest)
    except subprocess.CalledProcessError as e:
        log.warning("gsutil upload failed for %s: %s", dest, e.stderr.strip())


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
    bucket: str | None = None,
    gcs_prefix: str | None = None,
) -> dict[str, Path]:
    """Save train/val/test splits as parquet files.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.
        out_dir: Local output directory.
        bucket: GCS bucket name (optional).
        gcs_prefix: GCS path prefix (optional).

    Returns:
        Dict mapping split name to local path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        local_path = out_dir / f"{name}.parquet"
        split_df.to_parquet(local_path, index=False)
        paths[name] = local_path
        log.info("Saved %s split: %s (%d rows)", name, local_path, len(split_df))
        if bucket and gcs_prefix:
            _upload_to_gcs(local_path, bucket, f"{gcs_prefix}{name}.parquet")
    return paths


def save_scaler(
    pipeline: Pipeline,
    numeric_cols: list[str],
    out_dir: Path,
    bucket: str | None = None,
    gcs_path: str | None = None,
) -> Path:
    """Serialize the scaler pipeline and column list to pickle.

    Args:
        pipeline: Fitted sklearn Pipeline.
        numeric_cols: Feature column names.
        out_dir: Local output directory.
        bucket: GCS bucket name (optional).
        gcs_path: GCS object path (optional).

    Returns:
        Local path to the saved pickle.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / "scaler_pipeline.pkl"
    with open(local_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "columns": numeric_cols}, f)
    log.info("Saved scaler pipeline: %s", local_path)
    if bucket and gcs_path:
        _upload_to_gcs(local_path, bucket, gcs_path)
    return local_path


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run_splitting_pipeline(
    source: str | Path | None = None,
    cfg: Settings = settings,
    upload: bool = False,
    train_years: tuple[int, int] | None = None,
    val_years: tuple[int, int] | None = None,
    test_years: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Run the full splitting pipeline: load → split → scale → save.

    Args:
        source: Path to local CSV/parquet, or None for BigQuery.
        cfg: Settings instance.
        upload: Whether to upload to GCS.
        train_years: Override for training year range.
        val_years: Override for validation year range.
        test_years: Override for test year range.

    Returns:
        Dict with split paths, scaler path, and summary statistics.
    """
    ty = train_years or cfg.train_years
    vy = val_years or cfg.val_years
    tey = test_years or cfg.test_years

    out_dir = Path(cfg.local_splits_dir)
    bucket = cfg.gcs_bucket if upload else None
    gcs_prefix = cfg.splits_output_path if upload else None

    # 1. Load
    df = load_features(source, cfg)
    log.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    # 2. Build stratification key
    df = make_stratification_key(df)

    # 3. Time-based split
    train, val, test = time_based_split(
        df,
        train_years=ty,
        val_years=vy,
        test_years=tey,
        exclude_years=cfg.exclude_years,
    )

    # 4. Validate
    validate_no_temporal_leakage(train, val, test)
    validate_stratification(train, val, test)

    # 5. Fit scaler on training data only
    numeric_cols = get_numeric_columns(train)
    pipeline, numeric_cols = fit_scaler(train, numeric_cols)

    # 6. Apply scaler to all splits
    train_scaled = apply_scaler(train, pipeline, numeric_cols)
    val_scaled = apply_scaler(val, pipeline, numeric_cols)
    test_scaled = apply_scaler(test, pipeline, numeric_cols)

    # 7. Save splits
    split_paths = save_splits(
        train_scaled,
        val_scaled,
        test_scaled,
        out_dir,
        bucket,
        gcs_prefix,
    )

    # 8. Save scaler
    scaler_path = save_scaler(
        pipeline,
        numeric_cols,
        out_dir,
        bucket,
        cfg.scaler_output_path if upload else None,
    )

    result = {
        "split_paths": {k: str(v) for k, v in split_paths.items()},
        "scaler_path": str(scaler_path),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
    }
    log.info("Splitting pipeline complete: %s", json.dumps(result, indent=2))
    return result
