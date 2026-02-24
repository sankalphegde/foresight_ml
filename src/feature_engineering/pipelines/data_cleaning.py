"""Data Cleaning & NaN Handling Module
====================================
Handles all missing value imputation and column dropping BEFORE feature engineering.
Every decision is documented with rationale for reproducibility and audit.

Column-by-Column Strategy:
--------------------------
DROPPED:
  - EarningsPerShareBasic       (100% null — no data exists)
  - EarningsPerShareDiluted     (100% null — no data exists)
  - quality_check_flag          (single value "Valid" — zero information gain)

IMPUTED — Macroeconomic (65.8% null):
  - fed_funds, unemployment, inflation
    Strategy: Forward-fill by filed_date order, then backfill
    Rationale: These are time-indexed, economy-wide values (not company-specific).
    A fed funds rate of 1.25% persists until the next FOMC meeting changes it.
    Forward-fill propagates the most recent known value; backfill handles
    the earliest rows where no prior observation exists.

IMPUTED — Financial (0% null currently, but guarded for production):
  - Balance sheet items → Forward-fill within company, then fill 0
  - Income statement items → Forward-fill within company, then fill 0
  - Cash flow items → Fill with 0 (flows, not stocks)
"""

import logging
from typing import cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Column Classification ───────────────────────────────────────────────────

COLUMNS_TO_DROP = [
    "EarningsPerShareBasic",  # 100% null
    "EarningsPerShareDiluted",  # 100% null
    "quality_check_flag",  # Single value, no information gain
]

MACRO_COLUMNS = [
    "fed_funds",
    "unemployment",
    "inflation",
]

BALANCE_SHEET_COLUMNS = [
    "Assets",
    "AssetsCurrent",
    "CashAndCashEquivalentsAtCarryingValue",
    "InventoryNet",
    "AccountsReceivableNetCurrent",
    "PropertyPlantAndEquipmentNet",
    "Goodwill",
    "IntangibleAssetsNetExcludingGoodwill",
    "Liabilities",
    "LiabilitiesCurrent",
    "AccountsPayableCurrent",
    "LongTermDebt",
    "LongTermDebtCurrent",
    "StockholdersEquity",
    "RetainedEarningsAccumulatedDeficit",
    "AdditionalPaidInCapital",
    "CommonStockValue",
]

INCOME_STATEMENT_COLUMNS = [
    "Revenues",
    "CostOfGoodsAndServicesSold",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "ResearchAndDevelopmentExpense",
    "SellingGeneralAndAdministrativeExpense",
    "InterestExpense",
    "IncomeTaxExpenseBenefit",
]

CASH_FLOW_COLUMNS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities",
    "DepreciationDepletionAndAmortization",
]


def _log_null_summary(df: pd.DataFrame, stage: str) -> dict:
    """Log and return null counts per column for auditing."""
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    cols_with_nulls = null_counts[null_counts > 0]

    logger.info(f"[{stage}] Total NaN cells: {total_nulls}")
    if len(cols_with_nulls) > 0:
        for col, count in cols_with_nulls.items():
            pct = 100 * count / len(df)
            logger.info(f"  {col}: {count} nulls ({pct:.1f}%)")
    else:
        logger.info("  No nulls remaining.")

    return cast(dict, null_counts.to_dict())


def drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that carry no useful information.

    - EarningsPerShareBasic:   100% null — entirely missing, cannot impute.
    - EarningsPerShareDiluted: 100% null — entirely missing, cannot impute.
    - quality_check_flag:      Single unique value ("Valid") across all rows.
                               Provides zero discriminative power for any model.
    """
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    dropped = []

    for col in cols_present:
        if col in df.columns:
            null_pct = 100 * df[col].isnull().sum() / len(df)
            nunique = df[col].nunique()
            logger.info(f"Dropping '{col}': {null_pct:.1f}% null, {nunique} unique values")
            dropped.append(col)

    df = df.drop(columns=dropped, errors="ignore")
    logger.info(f"Dropped {len(dropped)} columns: {dropped}")
    return df


def impute_macro_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Impute macroeconomic columns using forward-fill by date, then backfill.

    Rationale:
    ----------
    Macroeconomic indicators (fed_funds, unemployment, inflation) are
    economy-wide values that change on known schedules:
      - Fed Funds Rate: changes at FOMC meetings (~8 times/year)
      - Unemployment: released monthly by BLS
      - Inflation (CPI): released monthly by BLS

    These values are NOT company-specific — all companies filing on the same
    date share the same macro environment. The 65.8% null pattern occurs
    because the source data join didn't match macro data for many rows.

    Forward-fill propagates the last known value (correct behavior — the rate
    stays constant until officially changed). Backfill handles the very
    earliest rows where no prior observation exists.
    """
    macro_cols_present = [c for c in MACRO_COLUMNS if c in df.columns]
    if not macro_cols_present:
        logger.warning("No macro columns found to impute.")
        return df

    # Sort by date to ensure correct temporal ordering for ffill
    df = df.sort_values("filed_date").reset_index(drop=True)

    imputation_counts = {}
    for col in macro_cols_present:
        before_nulls = df[col].isnull().sum()

        # Forward-fill: propagate the most recent known macro value
        df[col] = df[col].ffill()
        # Backfill: handle the earliest rows where no prior value exists
        df[col] = df[col].bfill()

        after_nulls = df[col].isnull().sum()
        imputed = before_nulls - after_nulls
        imputation_counts[col] = imputed
        logger.info(
            f"Imputed '{col}': {imputed} values (before={before_nulls}, after={after_nulls})"
        )

    return df


def impute_financial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Impute financial statement columns with company-specific forward-fill.

    Rationale:
    ----------
    Balance Sheet Items (Assets, Liabilities, Equity, etc.):
      → Forward-fill within each company, then fill 0.
      These are STOCK variables (point-in-time snapshots). If a company
      doesn't report a line item in quarter Q, the best estimate is the
      prior quarter's value (the balance persists). Zero fallback means
      "this item is not applicable to this company."

    Income Statement Items (Revenue, COGS, NetIncome, etc.):
      → Forward-fill within each company, then fill 0.
      Same logic — if a quarterly income item is missing, carry forward
      the last reported value as the best estimate.

    Cash Flow Items (Operating CF, Investing CF, Financing CF):
      → Fill with 0 directly (no forward-fill).
      Cash flows are FLOW variables (activity over a period). A missing
      cash flow item most likely means zero activity in that category
      for the period, not that the prior period's flow continued.
    """
    # Sort by company and time for correct forward-fill order
    # Support both old (cik) and new (firm_id) column names
    id_col = "firm_id" if "firm_id" in df.columns else "cik"
    df = df.sort_values([id_col, "fiscal_year", "fiscal_period"]).reset_index(drop=True)

    # Balance sheet: forward-fill within company, then 0
    bs_cols = [c for c in BALANCE_SHEET_COLUMNS if c in df.columns]
    for col in bs_cols:
        before_nulls = df[col].isnull().sum()
        df[col] = df.groupby(id_col)[col].ffill()
        df[col] = df[col].fillna(0)
        imputed = before_nulls - df[col].isnull().sum()
        if imputed > 0:
            logger.info(f"Imputed '{col}' (balance sheet): {imputed} values via ffill+0")

    # Income statement: forward-fill within company, then 0
    is_cols = [c for c in INCOME_STATEMENT_COLUMNS if c in df.columns]
    for col in is_cols:
        before_nulls = df[col].isnull().sum()
        df[col] = df.groupby(id_col)[col].ffill()
        df[col] = df[col].fillna(0)
        imputed = before_nulls - df[col].isnull().sum()
        if imputed > 0:
            logger.info(f"Imputed '{col}' (income stmt): {imputed} values via ffill+0")

    # Cash flow: fill with 0 directly
    cf_cols = [c for c in CASH_FLOW_COLUMNS if c in df.columns]
    for col in cf_cols:
        before_nulls = df[col].isnull().sum()
        df[col] = df[col].fillna(0)
        imputed = before_nulls - df[col].isnull().sum()
        if imputed > 0:
            logger.info(f"Imputed '{col}' (cash flow): {imputed} values via fill(0)")

    return df


def validate_post_cleaning(df: pd.DataFrame) -> None:
    """Post-imputation validation checks.
    Raises AssertionError if data quality issues remain.
    """
    # Check no infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        logger.error(f"Infinite values found: {inf_cols.to_dict()}")
        raise AssertionError(f"Infinite values in columns: {inf_cols.index.tolist()}")

    # Check dropped columns are gone
    for col in COLUMNS_TO_DROP:
        assert col not in df.columns, f"Column '{col}' should have been dropped"

    # Log final null summary
    _log_null_summary(df, "POST-CLEANING")
    logger.info("✓ Post-cleaning validation passed.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main entry point: run all cleaning steps in order.

    Pipeline:
      1. Log pre-cleaning null summary
      2. Drop uninformative columns (100% null + zero-variance)
      3. Impute macroeconomic columns (ffill by date + bfill)
      4. Impute financial columns (ffill within company + zero-fill)
      5. Validate post-cleaning state

    Returns the cleaned DataFrame.
    """
    logger.info(f"Starting data cleaning. Shape: {df.shape}")

    # Record pre-cleaning state for audit
    _log_null_summary(df, "PRE-CLEANING")

    # Step 1: Drop uninformative columns
    df = drop_uninformative_columns(df)

    # Step 2: Impute macro columns
    df = impute_macro_columns(df)

    # Step 3: Impute financial columns
    df = impute_financial_columns(df)

    # Step 4: Validate
    validate_post_cleaning(df)

    logger.info(f"Data cleaning complete. Final shape: {df.shape}")
    return df
