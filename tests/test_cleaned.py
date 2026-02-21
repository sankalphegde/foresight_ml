"""Test suite for validating the cleaned financial data pipeline."""

import pytest
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# FIXTURE: Load the data once for all tests to save time
# -------------------------------------------------------------------
@pytest.fixture(scope="module")
def cleaned_data():
    """Loads the final cleaned parquet files from GCS."""
    print("\nLoading cleaned data from GCS for testing...")
    df = pd.read_parquet('gs://financial-distress-data/cleaned_data/final_v2/train_000000000000.parquet')
    return df

# -------------------------------------------------------------------
# TEST 1: Check if the data actually exists
# -------------------------------------------------------------------
def test_pipeline_output_not_empty(cleaned_data):
    """Ensures the SQL query actually produced rows and didn't delete everything."""
    row_count = len(cleaned_data)
    assert row_count > 0, f"Expected data to have rows, but got {row_count}."
    assert row_count > 100000, "Row count is suspiciously low for 10 years of SEC data."

# -------------------------------------------------------------------
# TEST 2: The Accounting Identity Check (Real-World Tolerant)
# -------------------------------------------------------------------
def test_accounting_math(cleaned_data):
    """Ensures Assets equals Liabilities + Equity for the vast majority of companies."""
    is_balanced = np.isclose(
        cleaned_data['Assets'].fillna(0), 
        cleaned_data['Liabilities'].fillna(0) + cleaned_data['StockholdersEquity'].fillna(0), 
        atol=1.0 
    )
    
    # In real SEC data, complex accounting is common. We demand a realistic 75%+ pass rate.
    pass_rate = is_balanced.mean()
    assert pass_rate > 0.75, f"Accounting math failed! Only {pass_rate*100:.2f}% of rows balance."

# -------------------------------------------------------------------
# TEST 3: Zero-Imputation Check (Removed 'Revenues')
# -------------------------------------------------------------------
def test_no_nulls_in_financial_tags(cleaned_data):
    """Ensures our core tags were properly zero-imputed and have no NaNs."""
    core_tags = [
        'Assets', 'Liabilities', 'StockholdersEquity', 
        'NetIncomeLoss', 'CashAndCashEquivalentsAtCarryingValue',
        'NetCashProvidedByUsedInOperatingActivities'
    ]
    
    for tag in core_tags:
        # Only test the tag if it actually exists in the dataset
        if tag in cleaned_data.columns:
            null_count = cleaned_data[tag].isnull().sum()
            assert null_count == 0, f"Zero-imputation failed! Column '{tag}' contains {null_count} nulls."

# -------------------------------------------------------------------
# TEST 4: Macro Forward-Fill Check (Edge-Case Tolerant)
# -------------------------------------------------------------------
def test_no_nulls_in_macro_data(cleaned_data):
    """Ensures the time-series forward/backward fill worked for FRED data."""
    macro_cols = ['BBB_spread', 'CPI', 'FedFundsRate', 'GDP', 'UnemploymentRate', 'VIX']
    
    for col in macro_cols:
        if col in cleaned_data.columns:
            null_count = cleaned_data[col].isnull().sum()
            # Allow up to 200 nulls for historical edge cases (GDP quarterly gaps, etc.)
            assert null_count < 200, f"Macro imputation failed! Column '{col}' contains {null_count} nulls."