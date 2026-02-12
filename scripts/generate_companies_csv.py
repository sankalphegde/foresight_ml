import pandas as pd
import requests

# SEC canonical company ticker list
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

# Where to save your CSV
OUTPUT_PATH = "./data/companies.csv"
MAX_COMPANIES = 15000

# Pull data
headers = {"User-Agent": "foresight-ml research bryan@example.com"}
resp = requests.get(SEC_TICKER_URL, headers=headers)
resp.raise_for_status()

data = resp.json()

# Build DataFrame
rows = []
for _, entry in data.items():
    rows.append(
        {
            "ticker": entry["ticker"],
            "cik": entry["cik_str"],  # keep as integer string
        }
    )

df = pd.DataFrame(rows)
df = df.head(MAX_COMPANIES)

# Save CSV
df.to_csv(OUTPUT_PATH, index=False)

print(f"Wrote {len(df)} companies to {OUTPUT_PATH}")
