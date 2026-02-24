# Feature Engineering & Bias Analysis Pipeline

A scalable feature engineering and bias analysis pipeline for the Corporate Financial Distress Early-Warning System, running on GCP with BigQuery as the compute/storage backbone.

## Background

The project predicts corporate financial distress 6–12 months ahead using SEC EDGAR quarterly filings and FRED macroeconomic data. The dataset has **3,425 rows × 41 columns** covering **77 companies** from 2009–2026. The pipeline must:

1. Engineer ~45 features (financial ratios, growth rates, rolling statistics, interaction terms)
2. Run bias analysis across sector, company size, time period, and macroeconomic regime slices
3. Be BigQuery-native for scalability, with a local Pandas mode for development/testing

## User Review Required

> [!IMPORTANT]
> **GCP Configuration**: The pipeline needs your GCP project ID, BigQuery dataset name, and service account credentials path. These are parameterized in `config/settings.yaml` — you'll need to fill them in before running the BigQuery mode.

> [!IMPORTANT]
> **Distress Labels**: The current dataset does not contain the target column `DistressNext12Months`. The feature engineering pipeline creates features only — labeling is a separate step. Bias analysis will use the distress labels if present, otherwise it will analyze feature distributions across slices.

## Proposed Changes

### Project Structure

```
mlops/
├── README.md                         # Project overview, setup, usage guide
├── requirements.txt                  # Python dependencies
├── config/
│   └── settings.yaml                # GCP project, dataset, table config
├── pipelines/
│   ├── __init__.py
│   ├── data_cleaning.py             # NaN imputation/dropping with explanations
│   ├── feature_engineering.py        # Core feature engineering logic (Pandas)
│   ├── feature_engineering_bq.sql    # BigQuery SQL (cleaning + features)
│   ├── bias_analysis.py             # Bias analysis across slices
│   ├── visualizations.py            # All charts and plots
│   └── run_pipeline.py              # CLI orchestrator (local + BQ modes)
├── tests/
│   ├── __init__.py
│   ├── test_feature_engineering.py
│   └── test_bias_analysis.py
├── data/
│   └── cleaned_data_merged_v1_final_data_000000000000.parquet
└── documentation/
```

---

### README

#### [NEW] [README.md](file:///c:/Users/abamo/Desktop/coding/mlops/README.md)

Project README covering:
- **Overview**: What the project does, target variable, data sources
- **Setup**: Python dependencies (`pip install -r requirements.txt`), GCP auth (`gcloud auth application-default login`)
- **Usage — Local Mode**: `python -m pipelines.run_pipeline --mode local --input <parquet> --output data/`
- **Usage — BigQuery Mode**: `python -m pipelines.run_pipeline --mode bigquery --config config/settings.yaml`
- **Running Tests**: `python -m pytest tests/ -v`
- **Pipeline Outputs**: Description of `engineered_features.parquet` and `bias_report.csv`
- **Configuration**: How to edit `config/settings.yaml` for GCP
- **Feature Catalog**: Table listing all ~42 engineered features by category

---

### Config

#### [NEW] [settings.yaml](file:///c:/Users/abamo/Desktop/coding/mlops/config/settings.yaml)

YAML config with:
- `gcp.project_id`, `gcp.dataset`, `gcp.location` (user fills in)
- `tables.raw_financials`, `tables.engineered_features`, `tables.bias_report`
- `feature_engineering.rolling_windows` (default: `[4, 8]` quarters)
- `feature_engineering.growth_lag` (default: `4` quarters for YoY)

---

### Data Cleaning & NaN Handling

#### [NEW] [data_cleaning.py](file:///c:/Users/abamo/Desktop/coding/mlops/pipelines/data_cleaning.py)

Runs **before** feature engineering. Every column's NaN strategy is documented with rationale.

**Columns DROPPED entirely:**

| Column | Null % | Action | Rationale |
|---|---|---|---|
| `EarningsPerShareBasic` | 100% | **Drop** | Entirely null — no data available |
| `EarningsPerShareDiluted` | 100% | **Drop** | Entirely null — no data available |
| `quality_check_flag` | 0% | **Drop after validation** | Single unique value (`"Valid"`); no information gain |

**Columns IMPUTED — macroeconomic (65.8% null = 2,253 of 3,425 rows):**

| Column | Null % | Action | Rationale |
|---|---|---|---|
| `fed_funds` | 65.8% | **Forward-fill by date, then backfill** | Macro indicators are time-indexed, not company-specific. FFill uses the most recent known rate (e.g., the Fed rate doesn't change between FOMC meetings). Backfill handles the earliest rows where no prior value exists. |
| `unemployment` | 65.8% | **Forward-fill by date, then backfill** | Same logic — unemployment rates change monthly but are consistent across all companies at a given date |
| `inflation` | 65.8% | **Forward-fill by date, then backfill** | CPI values are released monthly; FFill propagates the latest reading |

> [!NOTE]
> The macro columns share the same null pattern (identical 2,253 rows are null). This is likely because these rows come from a data join where macro data wasn't matched. Forward-fill by `filed_date` order is the most appropriate strategy since these are time-series values that persist until updated.

**Columns IMPUTED — financial (0% null currently, but guarded for production):**

| Column Group | Action | Rationale |
|---|---|---|
| Balance sheet items (Assets, Liabilities, Equity, etc.) | **Forward-fill within company, then 0** | A missing balance sheet item likely means the company didn't report it — FFill assumes the prior period's value persists; 0 fallback means "not applicable" |
| Income statement items (Revenue, COGS, NetIncome, etc.) | **Forward-fill within company, then 0** | Same logic — income items carry forward as the best estimate for a missing quarter |
| Cash flow items (OperatingCF, InvestingCF, FinancingCF) | **Fill with 0** | Cash flow items are flow measures (not stocks); a missing value most likely means zero activity in that category |

**Post-imputation validation:**
- Log count of NaNs remaining per column (should be 0)
- Log imputation counts per column for audit trail
- Assert no infinite values introduced

**BigQuery equivalent**: The SQL script uses `LAST_VALUE(col IGNORE NULLS) OVER (...)` for forward-fill and `IFNULL(col, 0)` for zero-fill.

---

### Feature Engineering (Python — local mode)

#### [NEW] [feature_engineering.py](file:///c:/Users/abamo/Desktop/coding/mlops/pipelines/feature_engineering.py)

Core Pandas-based feature engineering module. Each function is pure and testable.

**Financial Ratios (13 features):**

| Feature | Formula | Category |
|---|---|---|
| `current_ratio` | AssetsCurrent / LiabilitiesCurrent | Liquidity |
| `quick_ratio` | (AssetsCurrent − InventoryNet) / LiabilitiesCurrent | Liquidity |
| `cash_ratio` | Cash / LiabilitiesCurrent | Liquidity |
| `debt_to_equity` | Liabilities / StockholdersEquity | Leverage |
| `debt_to_assets` | Liabilities / Assets | Leverage |
| `interest_coverage` | OperatingIncomeLoss / InterestExpense | Leverage |
| `gross_margin` | GrossProfit / Revenues | Profitability |
| `operating_margin` | OperatingIncomeLoss / Revenues | Profitability |
| `net_margin` | NetIncomeLoss / Revenues | Profitability |
| `roa` | NetIncomeLoss / Assets | Profitability |
| `roe` | NetIncomeLoss / StockholdersEquity | Profitability |
| `asset_turnover` | Revenues / Assets | Efficiency |
| `cash_flow_to_debt` | OperatingCashFlow / Liabilities | Cash Flow |

**Growth Rates (8 features):** YoY% change for Revenue, Assets, NetIncome, OperatingCashFlow, Liabilities, R&D, SGA, GrossProfit — computed per company via `groupby(cik).pct_change(4)`.

**Rolling Statistics (12 features):** 4Q and 8Q rolling mean + std for: `current_ratio`, `debt_to_equity`, `net_margin`, `roa`, `operating_cash_flow_to_debt`, `revenue_growth_yoy`.

**Z-Score & Interaction Terms (5 features):**
- `altman_z_approx`: Approximated Altman Z-score from available fields
- `cash_burn_rate`: (Cash − lag(Cash, 1)) / OperatingExpenses
- `leverage_x_margin`: debt_to_equity × operating_margin (stress interaction)
- `rd_intensity`: R&D / Revenues
- `sga_intensity`: SGA / Revenues

**Macro Interaction (3 features):**
- `fed_rate_x_leverage`: fed_funds × debt_to_equity
- `unemployment_x_margin`: unemployment × net_margin
- `inflation_x_cash_ratio`: inflation × cash_ratio

**Size Bucketing (1 feature):**
- `company_size_bucket`: Quartile-based bucket on Assets (`small`, `mid`, `large`, `mega`)

All denominators are guarded with `np.where(denom == 0, np.nan, ...)` to prevent division errors. Features are clipped to ±5 std to handle outliers.

---

### Feature Engineering (BigQuery SQL — scalable mode)

#### [NEW] [feature_engineering_bq.sql](file:///c:/Users/abamo/Desktop/coding/mlops/pipelines/feature_engineering_bq.sql)

A single BigQuery SQL script that mirrors the Python logic:
- Uses `SAFE_DIVIDE()` for all ratio computations
- Uses `LAG()` window functions partitioned by `cik` ordered by `fiscal_year, fiscal_period` for growth rates
- Uses `AVG() OVER (PARTITION BY cik ORDER BY ... ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)` for rolling stats
- Uses `NTILE(4)` for company size bucketing
- Creates or replaces table `{dataset}.engineered_features`

---

### Bias Analysis

#### [NEW] [bias_analysis.py](file:///c:/Users/abamo/Desktop/coding/mlops/pipelines/bias_analysis.py)

Analyzes feature distributions and (optionally) model performance across protected/sensitive slices:

**Slicing Dimensions:**
1. **Company Size** — `company_size_bucket` (small/mid/large/mega)
2. **Sector Proxy** — Derived from financial profile (high R&D = tech, high inventory = manufacturing, etc.)
3. **Time Period** — Pre-2016 vs Post-2016 (captures regulatory + economic regime shifts)
4. **Macro Regime** — High vs Low fed funds rate periods

**Analysis Per Slice:**
- Sample counts and class balance (if distress labels exist)
- Feature distribution statistics (mean, std, median, IQR per slice)
- Missing data rates per slice
- Feature drift between slices (Population Stability Index / Jensen-Shannon divergence)
- Outlier concentration per slice

**Output:**
- Summary DataFrame saved as `bias_report.csv`
- Console report with key findings highlighted

---

### Visualizations

#### [NEW] [visualizations.py](file:///c:/Users/abamo/Desktop/coding/mlops/pipelines/visualizations.py)

Generates all charts into `data/plots/` directory. Uses `matplotlib` + `seaborn`.

**Feature Distribution & Correlation Charts:**

| # | Chart | Description |
|---|---|---|
| 1 | **Correlation Heatmap** | Full correlation matrix of all engineered features, annotated, clustered |
| 2 | **Top-20 Correlation Bar Chart** | Top 20 most correlated feature pairs (absolute value) |
| 3 | **Feature Distribution Grid** | Histogram + KDE for every engineered feature (grid of subplots) |
| 4 | **Box Plot Grid** | Box plots for all financial ratios, grouped by category (liquidity, leverage, profitability, efficiency) |
| 5 | **Pairplot (Key Features)** | Scatter matrix for top 6 features by variance |
| 6 | **Missing Data Heatmap** | Pre-imputation NaN pattern heatmap (rows × columns) |
| 7 | **Missing Data Bar Chart** | Per-column null percentage bar chart |

**Bias Analysis Charts:**

| # | Chart | Description |
|---|---|---|
| 8 | **Slice Sample Counts** | Bar chart of sample sizes per slice (size bucket, sector, time period, macro regime) |
| 9 | **Feature Distribution by Size Bucket** | Overlaid KDE plots for key ratios split by company_size_bucket |
| 10 | **Feature Distribution by Time Period** | Pre-2016 vs Post-2016 distribution comparison (violin plots) |
| 11 | **Feature Distribution by Sector Proxy** | Box plots per sector proxy for key financial ratios |
| 12 | **Macro Regime Comparison** | Side-by-side distributions for high vs low fed_funds regime |
| 13 | **PSI Heatmap** | Population Stability Index across all feature × slice combinations |
| 14 | **Outlier Concentration by Slice** | Stacked bar chart showing % of outliers in each slice |
| 15 | **Missing Rate by Slice** | Heatmap of missing data rates per feature per slice (pre-imputation) |

---

### Pipeline Orchestration

#### [NEW] [run_pipeline.py](file:///c:/Users/abamo/Desktop/coding/mlops/pipelines/run_pipeline.py)

CLI entry point:

```bash
# Local mode (uses parquet + Pandas)
python -m pipelines.run_pipeline --mode local --input data/cleaned_data_merged_v1_final_data_000000000000.parquet --output data/

# BigQuery mode (runs SQL in BQ)
python -m pipelines.run_pipeline --mode bigquery --config config/settings.yaml
```

- `--mode local`: Loads parquet → runs `feature_engineering.py` → runs `bias_analysis.py` → saves outputs
- `--mode bigquery`: Reads config → executes `feature_engineering_bq.sql` via BigQuery client → pulls results for bias analysis

---

### Tests

#### [NEW] [test_feature_engineering.py](file:///c:/Users/abamo/Desktop/coding/mlops/tests/test_feature_engineering.py)

- Tests NaN imputation logic (forward-fill, zero-fill, column dropping)
- Tests each ratio function with known inputs/outputs
- Tests division-by-zero guarding returns NaN
- Tests growth rate computation with a small multi-period DataFrame
- Tests rolling statistics window sizes
- Tests outlier clipping

#### [NEW] [test_bias_analysis.py](file:///c:/Users/abamo/Desktop/coding/mlops/tests/test_bias_analysis.py)

- Tests slice creation (size buckets, sector proxy, time periods)
- Tests per-slice statistics computation
- Tests PSI calculation with known distributions
- Tests handling of missing labels gracefully

---

## Verification Plan

### Automated Tests

```bash
# Run all unit tests (from the mlops root directory)
python -m pytest tests/ -v
```

Expected: All tests pass.

### Local Pipeline Execution

```bash
# Run the full local pipeline end-to-end
python -m pipelines.run_pipeline --mode local --input data/cleaned_data_merged_v1_final_data_000000000000.parquet --output data/
```

Expected:
- `data/engineered_features.parquet` is created with ~45 new feature columns
- `data/bias_report.csv` is created with per-slice statistics
- `data/plots/` directory created with 15 PNG charts
- No errors, no unclipped infinite values in the features
- Zero NaN rows after imputation (except intentional NaN from safe-divide)

### BigQuery SQL Validation

```bash
# Dry-run the BQ SQL to check syntax (requires gcloud auth)
bq query --dry_run --use_legacy_sql=false < pipelines/feature_engineering_bq.sql
```

### Manual Verification

1. After running the local pipeline, open `data/engineered_features.parquet` and check:
   - Row count matches input (3,425 rows)
   - `EarningsPerShareBasic`, `EarningsPerShareDiluted`, `quality_check_flag` are dropped
   - `fed_funds`, `unemployment`, `inflation` have 0 nulls after imputation
   - No infinite values: `df.replace([np.inf, -np.inf], np.nan).isna().sum()`
2. Open `data/bias_report.csv` and verify slices have non-zero sample counts
3. Open `data/plots/` and verify all 15 charts are generated and readable
