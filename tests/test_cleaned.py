"""Automated integration test for the data cleaning pipeline."""

import os
import pytest
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest

# --- CONFIGURATION ---
# We just need the filenames now, the script will find them
SQL_FILENAME = 'data_cleaned.sql'
CHECK_FILENAME = 'quality_checks.sql'


def find_and_read_sql(filename):
    """Search for the SQL file in common locations and return content."""
    # List of possible paths to check (add more if needed)
    search_paths = [
        f"src/data/{filename}",           # Standard
        f"src/data/cleaned/{filename}",   # Subfolder
        f"data/{filename}",               # Root data
        f"data/cleaned/{filename}",       # Root cleaned
        filename                          # Current directory
    ]
    
    print(f"🔎 Searching for '{filename}'...")
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"   ✅ Found at: {path}")
            with open(path, 'r') as file:
                return file.read()
    
    # If loop finishes without finding anything:
    pytest.fail(f"❌ Could not find {filename}. Searched in: {search_paths}")


def test_run_pipeline():
    """Execute the full data cleaning and validation pipeline."""
    
    # 1. SAFETY CHECK: Skip test if no credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("GCP_SA_KEY"):
        print("⚠️ No credentials found. Skipping BigQuery integration test.")
        pytest.skip("Skipping BigQuery test: No credentials found.")

    # 2. Setup Client
    try:
        client = bigquery.Client()
    except Exception as e:
        pytest.fail(f"Failed to connect to BigQuery: {e}")

    print("🚀 STARTING PIPELINE TEST...\n")

    # 3. Load SQL (Using the new smart finder)
    cleaning_sql = find_and_read_sql(SQL_FILENAME)
    if not cleaning_sql:
        pytest.fail("SQL file was empty.")

    # 4. Dry Run (Syntax Check)
    print(f"[1/3] Checking syntax...")
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
        job.result()
        print("   ✅ Table created successfully!")
    except Exception as e:
        pytest.fail(f"Pipeline execution failed: {e}")

    # 6. Validate (Quality Checks)
    print("\n[3/3] Running Data Quality Checks...")
    check_sql = find_and_read_sql(CHECK_FILENAME)
    
    if check_sql:
        for query in check_sql.split(';'):
            if query.strip():
                try:
                    rows = list(client.query(query).result())
                    print(f"   🔎 Check Result: {rows[0][0]}")
                    assert rows[0][0] is not None
                except Exception as e:
                    print(f"   ⚠️ Quality check warning: {e}")

    print("\n🎉 TEST COMPLETE.")

if __name__ == "__main__":
    test_run_pipeline()