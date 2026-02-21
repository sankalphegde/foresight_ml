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
# TEST 2: The Accounting Identity Check
# -------------------------------------------------------------------
def test_accounting_math(cleaned_data):
    """Ensures Assets exactly equals Liabilities + StockholdersEquity."""
    # We use np.isclose instead of `==` to prevent tiny floating-point rounding errors from failing the test
    is_balanced = np.isclose(
        cleaned_data['Assets'], 
        cleaned_data['Liabilities'] + cleaned_data['StockholdersEquity'], 
        atol=1.0 # Allow a $1 rounding difference
    )
    
    # Assert that ALL rows are balanced
    assert is_balanced.all(), "Accounting identity (Assets = Liabilities + Equity) failed!"

# -------------------------------------------------------------------
# TEST 3: Zero-Imputation Check (No Nulls in Core Financials)
# -------------------------------------------------------------------
def test_no_nulls_in_financial_tags(cleaned_data):
    """Ensures our 40 tags were properly zero-imputed and have no NaNs."""
    core_tags = [
        'Assets', 'Liabilities', 'StockholdersEquity', 
        'NetIncomeLoss', 'Revenues', 'CashAndCashEquivalentsAtCarryingValue',
        'NetCashProvidedByUsedInOperatingActivities'
        # You can add the rest of your 40 tags here if you want to be strict!
    ]
    
    for tag in core_tags:
        null_count = cleaned_data[tag].isnull().sum()
        assert null_count == 0, f"Zero-imputation failed! Column '{tag}' contains {null_count} nulls."

# -------------------------------------------------------------------
# TEST 4: Macro Forward-Fill Check
# -------------------------------------------------------------------
def test_no_nulls_in_macro_data(cleaned_data):
    """Ensures the time-series forward/backward fill worked for FRED data."""
    macro_cols = ['BBB_spread', 'CPI', 'FedFundsRate', 'GDP', 'UnemploymentRate', 'VIX']
    
    for col in macro_cols:
        null_count = cleaned_data[col].isnull().sum()
        assert null_count == 0, f"Macro imputation failed! Column '{col}' contains {null_count} nulls."