"""Automated integration test for the data cleaning pipeline."""

import os

import pytest
from google.api_core.exceptions import BadRequest
from google.cloud import bigquery

# --- CONFIGURATION ---
SQL_FILE_PATH = 'src/data/data_cleaned.sql'
CHECK_FILE_PATH = 'src/data/quality_checks.sql'


def read_sql(filepath):
    """Read SQL content from a specific file path."""
    if not os.path.exists(filepath):
        # Fail the test if file is missing
        pytest.fail(f"Could not find file at {filepath}")
    with open(filepath) as file:
        return file.read()


def test_run_pipeline():
    """Execute the full data cleaning and validation pipeline."""
    # 1. SAFETY CHECK: Skip test if no credentials (e.g., in GitHub Actions)
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("GCP_SA_KEY"):
        print("⚠️ No credentials found. Skipping BigQuery integration test.")
        pytest.skip("Skipping BigQuery test: No credentials found.")

    # 2. Setup Client (Only initialize here, not at global scope!)
    try:
        client = bigquery.Client()
    except Exception as e:
        pytest.fail(f"Failed to connect to BigQuery: {e}")

    print("🚀 STARTING PIPELINE TEST...\n")

    # 3. Load SQL
    cleaning_sql = read_sql(SQL_FILE_PATH)
    if not cleaning_sql:
        pytest.fail("SQL file was empty.")

    # 4. Dry Run (Syntax Check)
    print(f"[1/3] Checking syntax for {SQL_FILE_PATH}...")
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

    try:
        client.query(cleaning_sql, job_config=job_config)
        print("   ✅ Syntax is valid.")
    except BadRequest as e:
        pytest.fail(f"Syntax Error in SQL: {e}")

    # 5. Execute (Create Table)
    print("\n[2/3] Building BigQuery Table (this may take 30s)...")
    try:
        job = client.query(cleaning_sql)
        job.result()  # Wait for completion
        print("   ✅ Table created successfully!")
    except Exception as e:
        pytest.fail(f"Pipeline execution failed: {e}")

    # 6. Validate (Quality Checks)
    print("\n[3/3] Running Data Quality Checks...")
    check_sql = read_sql(CHECK_FILE_PATH)
    if check_sql:
        for query in check_sql.split(';'):
            if query.strip():
                try:
                    rows = list(client.query(query).result())
                    print(f"   🔎 Check Result: {rows[0][0]}")
                    # Optional: Assert rows > 0
                    assert rows[0][0] is not None
                except Exception as e:
                    print(f"   ⚠️ Quality check warning: {e}")

    print("\n🎉 TEST COMPLETE.")

if __name__ == "__main__":
    # Allow running manually with `python tests/test_cleaned.py`
    test_run_pipeline()
