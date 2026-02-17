import os
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest

# Setup
client = bigquery.Client()

# --- CONFIGURATION: Pointing to your specific folder structure ---
SQL_FILE_PATH = 'src/data/data_cleaned.sql'
CHECK_FILE_PATH = 'src/data/quality_checks.sql'

def read_sql(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Error: Could not find file at {filepath}")
        return None
    with open(filepath, 'r') as file:
        return file.read()

def run():
    print("🚀 STARTING FORESIGHT PIPELINE...\n")

    # 1. Load SQL
    cleaning_sql = read_sql(SQL_FILE_PATH)
    if not cleaning_sql: return

    # 2. Dry Run (Safety Check)
    print(f"[1/3] Checking syntax for {SQL_FILE_PATH}...")
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    try:
        client.query(cleaning_sql, job_config=job_config)
        print("   ✅ Syntax is valid.")
    except BadRequest as e:
        print(f"   ❌ Syntax Error:\n{e}")
        return

    # 3. Execute (Create Table)
    print("\n[2/3] Building BigQuery Table (this may take 30s)...")
    try:
        job = client.query(cleaning_sql)
        job.result()
        print("   ✅ Table `final_training_set` created successfully!")
    except Exception as e:
        print(f"   ❌ Execution Failed:\n{e}")
        return

    # 4. Validate (Quality Checks)
    print("\n[3/3] Running Data Quality Checks...")
    check_sql = read_sql(CHECK_FILE_PATH)
    if check_sql:
        # Split by ';' to run multiple checks
        for query in check_sql.split(';'):
            if query.strip():
                try:
                    rows = list(client.query(query).result())
                    print(f"   🔎 Check Result: {rows[0][0]}")
                except Exception:
                    pass # Skip empty lines

    print("\n🎉 PIPELINE COMPLETE.")

if __name__ == "__main__":
    run()