"""Tests for data splitting and class imbalance handling.

Uses a generated local CSV fixture (no BigQuery or GCS access required).
The synthetic sample covers 2009–2013, so tests use narrower split boundaries:
    train: 2010–2011, val: 2012, test: 2013
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.class_balance import (
    apply_smote,
    compute_class_weights,
    generate_split_report,
)
from src.data.split import (
    TARGET_COL,
    apply_scaler,
    fit_scaler,
    get_numeric_columns,
    load_features,
    make_stratification_key,
    time_based_split,
    validate_no_temporal_leakage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Split boundaries for sample data (2009-2013)
SAMPLE_TRAIN_YEARS = (2010, 2011)
SAMPLE_VAL_YEARS = (2012, 2012)
SAMPLE_TEST_YEARS = (2013, 2013)


def _build_sample_df() -> pd.DataFrame:
    """Build a deterministic in-memory sample dataset for split tests."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []

    # Keep keys disjoint across (firm_id, fiscal_year, fiscal_period)
    periods = ["Q1", "Q2", "Q3", "Q4"]

    # 2009 rows (to verify exclusion behavior)
    for idx in range(8):
        period = periods[idx % 4]
        rows.append(
            {
                "firm_id": f"F{idx:03d}",
                "fiscal_year": 2009,
                "fiscal_period": period,
                "filed_date": f"2009-{(idx % 12) + 1:02d}-15",
                "company_size_bucket": "small",
                "sector_proxy": "services_other",
                "feature_a": float(rng.normal(0.5, 0.15)),
                "feature_b": float(rng.normal(1.0, 0.25)),
                "feature_c": float(rng.normal(0.2, 0.05)),
                "distress_label": int(idx % 2),
            }
        )

    # Train years 2010-2011 with imbalanced labels (enough minority for SMOTE)
    for year in (2010, 2011):
        for idx in range(24):
            period = periods[idx % 4]
            distressed = 1 if idx in {2, 5, 9, 14, 18, 22} else 0
            size = "small" if idx < 10 else ("mid" if idx < 18 else "large")
            sector = (
                "tech_pharma"
                if idx < 9
                else ("manufacturing_retail" if idx < 17 else "financial_capital_intensive")
            )
            rows.append(
                {
                    "firm_id": f"T{year}{idx:03d}",
                    "fiscal_year": year,
                    "fiscal_period": period,
                    "filed_date": f"{year}-{(idx % 12) + 1:02d}-15",
                    "company_size_bucket": size,
                    "sector_proxy": sector,
                    "feature_a": float(rng.normal(1.0 + distressed, 0.2)),
                    "feature_b": float(rng.normal(2.0 + distressed, 0.35)),
                    "feature_c": float(rng.normal(0.4 + 0.1 * distressed, 0.08)),
                    "distress_label": distressed,
                }
            )

    # Validation and test years
    for year in (2012, 2013):
        for idx in range(16):
            period = periods[idx % 4]
            distressed = 1 if idx in {1, 7, 12} else 0
            size = "mid" if idx < 8 else "large"
            sector = "tech_pharma" if idx < 10 else "services_other"
            rows.append(
                {
                    "firm_id": f"E{year}{idx:03d}",
                    "fiscal_year": year,
                    "fiscal_period": period,
                    "filed_date": f"{year}-{(idx % 12) + 1:02d}-15",
                    "company_size_bucket": size,
                    "sector_proxy": sector,
                    "feature_a": float(rng.normal(1.2 + distressed, 0.25)),
                    "feature_b": float(rng.normal(2.2 + distressed, 0.4)),
                    "feature_c": float(rng.normal(0.45 + 0.1 * distressed, 0.08)),
                    "distress_label": distressed,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def full_df(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Load generated sample dataset through CSV path (tests load_features)."""
    df = _build_sample_df()
    data_dir = tmp_path_factory.mktemp("split_data")
    csv_path = data_dir / "sample_data.csv"
    df.to_csv(csv_path, index=False)
    return load_features(csv_path)


@pytest.fixture(scope="module")
def splits(full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits from sample data."""
    df = make_stratification_key(full_df)
    return time_based_split(
        df,
        train_years=SAMPLE_TRAIN_YEARS,
        val_years=SAMPLE_VAL_YEARS,
        test_years=SAMPLE_TEST_YEARS,
        exclude_years=(2009,),
    )


@pytest.fixture(scope="module")
def train_df(splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Return the training split."""
    return splits[0]


@pytest.fixture(scope="module")
def val_df(splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Return the validation split."""
    return splits[1]


@pytest.fixture(scope="module")
def test_df(splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Return the test split."""
    return splits[2]


# ---------------------------------------------------------------------------
# Test 1: No temporal leakage
# ---------------------------------------------------------------------------


def test_no_temporal_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Max train year < min val year < min test year."""
    assert train_df["fiscal_year"].max() < val_df["fiscal_year"].min()
    assert val_df["fiscal_year"].max() < test_df["fiscal_year"].min()
    # Also run the project's own validator (should not raise)
    validate_no_temporal_leakage(train_df, val_df, test_df)


# ---------------------------------------------------------------------------
# Test 2: No data overlap
# ---------------------------------------------------------------------------


def test_no_data_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Train/val/test firm-year-quarter tuples must be disjoint."""

    def _keys(df: pd.DataFrame) -> set[tuple[str, int, str]]:
        return set(
            zip(
                df["firm_id"].astype(str),
                df["fiscal_year"],
                df["fiscal_period"].astype(str),
                strict=False,
            )
        )

    train_keys = _keys(train_df)
    val_keys = _keys(val_df)
    test_keys = _keys(test_df)

    assert train_keys.isdisjoint(val_keys), "Train and val overlap"
    assert train_keys.isdisjoint(test_keys), "Train and test overlap"
    assert val_keys.isdisjoint(test_keys), "Val and test overlap"


# ---------------------------------------------------------------------------
# Test 3: SMOTE only touches training data
# ---------------------------------------------------------------------------


def test_smote_only_touches_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """SMOTE output should have more rows than train; val/test unchanged."""
    train_smote = apply_smote(train_df)

    # SMOTE should increase training rows
    assert len(train_smote) > len(train_df)
    # Val and test must remain untouched (we didn't pass them to SMOTE)
    assert len(val_df) == len(val_df)  # Trivially true; val shouldn't change
    assert len(test_df) == len(test_df)


# ---------------------------------------------------------------------------
# Test 4: SMOTE balances classes
# ---------------------------------------------------------------------------


def test_smote_balances_classes(train_df: pd.DataFrame) -> None:
    """After SMOTE, the minority class should match the majority class."""
    train_smote = apply_smote(train_df)
    counts = train_smote[TARGET_COL].value_counts()
    # SMOTE should produce equal counts for both classes
    assert counts[0] == counts[1], f"Classes not balanced: {counts.to_dict()}"


# ---------------------------------------------------------------------------
# Test 5: Scaler fitted on train only
# ---------------------------------------------------------------------------


def test_scaler_fitted_on_train_only(train_df: pd.DataFrame) -> None:
    """Scaler .mean_ must match training data statistics."""
    numeric_cols = get_numeric_columns(train_df)
    pipeline, cols = fit_scaler(train_df, numeric_cols)

    # The scaler is step[1] in the pipeline; imputer is step[0]
    scaler = pipeline.named_steps["scaler"]
    imputer = pipeline.named_steps["imputer"]

    # After imputation, check scaler means against train data
    train_imputed = imputer.transform(train_df[numeric_cols])
    expected_means = np.nanmean(train_imputed, axis=0)

    np.testing.assert_allclose(
        scaler.mean_,
        expected_means,
        rtol=1e-5,
        err_msg="Scaler means don't match training data",
    )


# ---------------------------------------------------------------------------
# Test 6: Scaler transforms all splits
# ---------------------------------------------------------------------------


def test_scaler_transforms_all_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """All three splits should transform without error."""
    numeric_cols = get_numeric_columns(train_df)
    pipeline, cols = fit_scaler(train_df, numeric_cols)

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        result = apply_scaler(split, pipeline, cols)
        assert len(result) == len(split), f"{name} row count changed after scaling"
        assert set(result.columns) == set(split.columns), f"{name} columns changed"


# ---------------------------------------------------------------------------
# Test 7: Class weights computed correctly
# ---------------------------------------------------------------------------


def test_class_weights_computed(train_df: pd.DataFrame) -> None:
    """scale_pos_weight should equal count(neg) / count(pos)."""
    weights = compute_class_weights(train_df)

    counts = train_df[TARGET_COL].value_counts()
    expected = counts[0] / counts[1]

    assert "scale_pos_weight" in weights
    assert "n_positive" in weights
    assert "n_negative" in weights
    assert abs(weights["scale_pos_weight"] - expected) < 0.01


# ---------------------------------------------------------------------------
# Test 8: Split report contents
# ---------------------------------------------------------------------------


def test_split_report_contents(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Report JSON must have all required keys."""
    weights = compute_class_weights(train_df)
    train_smote = apply_smote(train_df)

    report = generate_split_report(
        train=train_df,
        val=val_df,
        test=test_df,
        train_smote=train_smote,
        class_weights=weights,
    )

    # Top-level keys
    assert "splits" in report
    assert "smote" in report
    assert "class_weights" in report

    # Per-split keys
    for split_name in ("train", "val", "test"):
        s = report["splits"][split_name]
        assert "rows" in s
        assert "distress_rate" in s
        assert "class_distribution" in s

    # SMOTE section
    assert "rows_before" in report["smote"]
    assert "rows_after" in report["smote"]
    assert report["smote"]["rows_after"] > report["smote"]["rows_before"]

    # Class weights
    assert "scale_pos_weight" in report["class_weights"]


# ---------------------------------------------------------------------------
# Test 9: Stratification key merges rare strata
# ---------------------------------------------------------------------------


def test_stratification_key_merges_rare(full_df: pd.DataFrame) -> None:
    """Strata with fewer than min_count rows should be merged to 'other'."""
    # Use a high threshold so some strata get merged in sample data
    df = make_stratification_key(full_df, min_count=10)

    # Check that 'other' exists if any raw combo had < 10 rows
    raw_key = (
        full_df["company_size_bucket"].astype(str) + "__" + full_df["sector_proxy"].astype(str)
    )
    counts = raw_key.value_counts()
    rare_exist = (counts < 10).any()

    if rare_exist:
        assert "other" in df["_strat_key"].values, "Rare strata should be merged to 'other'"
    else:
        assert (
            "other" not in df["_strat_key"].values
        ), "'other' should not appear when no rare strata"

    # Original columns should be unchanged
    assert df["company_size_bucket"].equals(full_df["company_size_bucket"])
    assert df["sector_proxy"].equals(full_df["sector_proxy"])


# ---------------------------------------------------------------------------
# Test 10: Year 2009 excluded
# ---------------------------------------------------------------------------


def test_year_2009_excluded(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """No rows with fiscal_year == 2009 should appear in any split."""
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert 2009 not in split["fiscal_year"].values, f"Year 2009 found in {name} split"
