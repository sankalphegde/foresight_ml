# Corporate Financial Distress — Feature Engineering & Bias Analysis Pipeline

A scalable pipeline that engineers **~42 financial features** from SEC EDGAR quarterly filings and FRED macroeconomic data, then runs **bias analysis** across company size, sector, time period, and macroeconomic regime slices. Runs both locally (Pandas) and at scale (BigQuery).

## Project Structure

```
mlops/
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml               # GCP project, dataset, feature params
├── pipelines/
│   ├── data_cleaning.py             # NaN handling (imputation + dropping)
│   ├── feature_engineering.py       # 42 features (Pandas)
│   ├── feature_engineering_bq.sql   # Same features (BigQuery SQL)
│   ├── bias_analysis.py             # Fairness analysis across slices
│   ├── visualizations.py            # 15 charts (distributions, bias, PSI)
│   └── run_pipeline.py              # CLI orchestrator
├── tests/
│   ├── test_feature_engineering.py
│   └── test_bias_analysis.py
└── data/
    └── cleaned_data_merged_v1_*.parquet
```

## Setup

**Python Dependencies:**
```bash
pip install -r requirements.txt
```

**GCP (for BigQuery mode only):**
```bash
gcloud auth application-default login
```
Then edit `config/settings.yaml` and set your `project_id` and `dataset`.

## Usage

### Local Mode (Pandas)
```bash
python -m pipelines.run_pipeline --mode local \
  --input data/cleaned_data_merged_v1_final_data_000000000000.parquet \
  --output data/
```

### BigQuery Mode
```bash
python -m pipelines.run_pipeline --mode bigquery --config config/settings.yaml
```

### Run Tests
```bash
python -m pytest tests/test_feature_engineering/ -v
```

## Pipeline Stages

| Stage | Module | What It Does |
|-------|--------|-------------|
| 1. Data Cleaning | `data_cleaning.py` | Drops 100%-null columns, forward-fills macro indicators, guards financial columns |
| 2. Feature Engineering | `feature_engineering.py` | Computes 42 features: ratios, growth rates, rolling stats, interactions |
| 3. Bias Analysis | `bias_analysis.py` | Analyzes distributions across 4 slicing dimensions with PSI/JS divergence |
| 4. Visualizations | `visualizations.py` | Generates 15 PNG charts to `data/plots/` |

## NaN Handling Strategy

| Column(s) | Action | Rationale |
|-----------|--------|-----------|
| `EarningsPerShareBasic/Diluted` | **Drop** | 100% null |
| `quality_check_flag` | **Drop** | Single value ("Valid"), zero info gain |
| `fed_funds, unemployment, inflation` | **FFill by date + backfill** | Macro values persist until updated |
| Balance sheet items | **FFill within company → 0** | Stock variables carry forward |
| Income statement items | **FFill within company → 0** | Best estimate for missing quarter |
| Cash flow items | **Fill 0** | Flows default to zero activity |

## Feature Catalog

| Category | Features | Count |
|----------|----------|-------|
| **Liquidity** | current_ratio, quick_ratio, cash_ratio | 3 |
| **Leverage** | debt_to_equity, debt_to_assets, interest_coverage | 3 |
| **Profitability** | gross_margin, operating_margin, net_margin, roa, roe | 5 |
| **Efficiency** | asset_turnover, cash_flow_to_debt | 2 |
| **Growth (YoY)** | revenue, assets, net_income, operating_cf, liabilities, R&D, SGA, gross_profit | 8 |
| **Rolling Stats** | 4Q & 8Q mean + std for 6 key features | 24 |
| **Composite** | altman_z_approx, cash_burn_rate, leverage×margin, rd_intensity, sga_intensity | 5 |
| **Macro Interaction** | fed_rate×leverage, unemployment×margin, inflation×cash_ratio | 3 |
| **Categorical** | company_size_bucket, sector_proxy | 2 |

## Outputs

After running the pipeline:

- **`data/engineered_features.parquet`** — Full dataset with all 42+ engineered features
- **`data/bias_report.csv`** — Per-slice statistics across 4 bias dimensions
- **`data/plots/`** — 15 PNG visualization files:
  - `01-07`: Correlation heatmap, top pairs, distributions, box plots, pairplot, missing data
  - `08-15`: Slice counts, size/time/sector/macro distributions, PSI heatmap, outlier concentration

## Configuration

All parameters are in `config/settings.yaml`:

```yaml
feature_engineering:
  rolling_windows: [4, 8]       # Rolling window sizes (quarters)
  growth_lag: 4                  # YoY growth lag (4 quarters)
  outlier_clip_std: 5            # Clip beyond ±5 std

bias_analysis:
  time_split_year: 2016          # Pre/post temporal split
  fed_funds_threshold: 2.0       # High/low rate regime split
```
