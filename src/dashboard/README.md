# Foresight-ML Dashboard

Interactive Streamlit dashboard for monitoring corporate financial distress predictions, model health, and data pipeline status.

## Quick Start

```bash
# From the project root
PYTHONPATH=. streamlit run src/dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. On first load, a branded splash screen plays while data is cached (~6 seconds).

### Prerequisites

- Python 3.10+
- All project dependencies installed (`pip install -e .`)
- One of the following data sources:
  - **GCS bucket** (`financial-distress-data`) with valid Google Cloud credentials
  - **Local artifacts** in `artifacts/` (model, splits, SHAP values)

## Architecture

```
src/dashboard/
├── app.py                  # Entry point — page config, sidebar nav, routing, splash screen
├── utils.py                # Shared helpers — risk classification, formatting, chart theme
├── data/
│   ├── gcs_loader.py       # Data layer — GCS + local fallback, cached with 5-min TTL
│   └── api_client.py       # Optional FastAPI client (falls back to GCS if unavailable)
└── pages/
    ├── company_risk.py     # Page 1 — Company Risk Explorer
    ├── watchlist.py        # Page 2 — High-Risk Watchlist
    ├── model_health.py     # Page 3 — Model Health
    └── pipeline_status.py  # Page 4 — Pipeline Status
```

## Pages

### 1. Company Risk Explorer (`company_risk.py`)

Search any company by name, ticker, or CIK number to view its predicted distress probability.

**Features:**

- Searchable selectbox with placeholder (starts empty, clearable with backspace)
- Risk badge: 🟢 Low (<30%), 🟡 Medium (30–70%), 🔴 High (>70%)
- Signal chips showing key financial health indicators (profitability, cash flow, leverage)
- Top 5 SHAP risk drivers with horizontal bar visualization and direction labels
- Quarterly trend chart (Plotly line chart with threshold lines at 0.30 and 0.70)
- Financial snapshot panel with formatted dollar amounts
- Fallback to binary distress labels when model predictions are unavailable
- Data freshness alert when predictions are from an older fiscal year
- Data quality check that warns about missing probability scores

### 2. High-Risk Watchlist (`watchlist.py`)

Ranked table of all scored companies filtered by predicted distress probability.

**Features:**

- Threshold slider with tooltip (default ≥ 0.50)
- Sector and company size filters
- CSV export button for filtered results
- Summary metrics: companies shown, high risk count, medium risk count, total scored
- Sector breakdown donut chart (appears when sector data is available)
- Quarter-over-quarter trend indicators (🔴 increased, 🟢 decreased)
- Active distress signal column
- Empty state guidance when threshold is set too high (>0.90)

### 3. Model Health (`model_health.py`)

Production model card, drift monitoring, and prediction quality overview.

**Features:**

- Drift alert banner (warning when drift detected, success when clear)
- Two-column layout: model card (version, ROC-AUC, hyperparameters, MLflow link) and drift monitor (PSI values, drifted features)
- Prediction distribution histogram (50 bins) with threshold lines
- Summary metrics with tooltips: total scored, mean probability, high risk count, median
- Per-slice performance table (ROC-AUC, recall, precision, Brier score by dimension)

### 4. Pipeline Status (`pipeline_status.py`)

Data ingestion and model training pipeline task status.

**Features:**

- Two-column layout: data pipeline (@daily, 8 tasks) and training pipeline (@weekly, 6 tasks)
- Color-coded status dots: 🟢 success, 🟡 warning, 🔴 failed, 🔵 running
- Real task status derived from artifact presence (model file, test split, SHAP values, predictions)
- Summary metrics with tooltips: last scored timestamp, companies scored, high-risk count, ROC-AUC
- Artifact status grid showing which pipeline outputs exist
- Links to MLflow, GCS Console, and GitHub

## Data Layer (`data/gcs_loader.py`)

All data loading is centralized in `gcs_loader.py` with Streamlit's `@st.cache_data` decorator (TTL = 300 seconds).

**Loading strategy (in priority order):**

| Source                              | When Used                                                |
| ----------------------------------- | -------------------------------------------------------- |
| GCS batch scores (`scores.parquet`) | Primary — from Person 5's batch inference                |
| Local model + test split            | Fallback — live `predict_proba` using local `artifacts/` |
| GCS model + test split              | Fallback — downloads model from GCS, scores on the fly   |
| Empty DataFrame                     | Graceful failure — dashboard shows appropriate warnings  |

**Data files loaded:**

| File              | GCS Path                                               | Description                                   |
| ----------------- | ------------------------------------------------------ | --------------------------------------------- |
| Scores            | `inference/scores_v1/scores.parquet`                   | Batch predictions with `distress_probability` |
| SHAP values       | `shap/shap_values.parquet`                             | Precomputed SHAP values per company-quarter   |
| Labeled panel     | `features/labeled_v1/labeled_panel.parquet`            | All company-quarter rows with financials      |
| Manifest          | `inference/scores_v1/manifest.json`                    | Model metadata, timestamps, ROC-AUC           |
| Optuna results    | `models/optuna_results.json`                           | Best hyperparameters, trial results           |
| Drift summary     | `monitoring/drift_reports/summary_latest.json`         | Latest PSI drift report                       |
| Slice performance | `mlflow/artifacts/slice_metrics/slice_performance.csv` | Per-slice evaluation metrics                  |
| Company names     | Local: `artifacts/reference/company_names.csv`         | CIK → ticker → name mapping                   |

## Shared Utilities (`utils.py`)

| Function                            | Description                                |
| ----------------------------------- | ------------------------------------------ |
| `risk_level(score)`                 | Returns `"High"` / `"Medium"` / `"Low"`    |
| `risk_emoji(score)`                 | Returns 🔴 / 🟡 / 🟢                       |
| `risk_color(score)`                 | Returns hex color string                   |
| `risk_badge_html(score)`            | Inline HTML badge with color and label     |
| `parse_top_features_json(json_str)` | Parses SHAP `top_features_json` column     |
| `shap_color(value)`                 | Red (increases risk) or green (protective) |
| `fmt_large_number(value)`           | Formats as `$24.2B`, `$3.1M`, `$500K`      |
| `quarter_label(year, period)`       | Returns `"Q3 2025"`                        |
| `quarter_sort_key(year, period)`    | Numeric key for chronological sorting      |
| `apply_chart_theme(fig)`            | Applies consistent Plotly styling          |

**Risk thresholds:** High ≥ 0.70, Medium ≥ 0.30, Low < 0.30

## API Client (`data/api_client.py`)

Optional REST client for the FastAPI service (`FORESIGHT_API_URL` env var). Falls back gracefully when the API is not deployed — the dashboard uses GCS data instead.

**Endpoints:** `/health`, `/model/info`, `/predict`, `/company/{cik}`, `/alerts`, `/drift/status`

## UI/UX Features

| Feature            | Implementation                                                            |
| ------------------ | ------------------------------------------------------------------------- |
| Loading splash     | Animated 6-second splash with progress bar and step indicators (`app.py`) |
| Sidebar navigation | Custom CSS radio buttons styled as text links with section headers        |
| Tooltips           | `help` parameter on all `st.metric`, `st.slider`, `st.selectbox` widgets  |
| How-to guides      | `st.expander("ℹ️ How to use this page")` on every page                    |
| Data alerts        | Staleness warning, drift banner, missing data quality check               |
| Empty states       | Friendly messages with actionable suggestions                             |
| Error handling     | User-friendly error messages with troubleshooting steps                   |
| Chart theme        | Consistent font, colors, grid styling via `apply_chart_theme()`           |
| Caching            | All loaders use `@st.cache_data(ttl=300)` for performance                 |
| Responsive layout  | `layout="wide"` with `st.columns` for side-by-side content                |

## Environment Variables

| Variable                         | Default                                         | Description                     |
| -------------------------------- | ----------------------------------------------- | ------------------------------- |
| `FORESIGHT_API_URL`              | `https://foresight-api-6ool3rlbea-uc.a.run.app` | FastAPI service URL             |
| `GOOGLE_APPLICATION_CREDENTIALS` | —                                               | Path to GCS service account key |

## Dependencies

The dashboard uses these packages (all in `pyproject.toml`):

- `streamlit` — UI framework
- `plotly` — Interactive charts
- `pandas` / `numpy` — Data manipulation
- `xgboost` — Live scoring fallback
- `google-cloud-storage` — GCS data access
- `requests` — API client
- `shap` — SHAP value computation (used by `src/models/explain.py`)
- `pyarrow` — Parquet file I/O

## Integration with Other Components

This dashboard is **Person 5's** deliverable and consumes outputs from:

| Person   | What They Produce                                    | How Dashboard Uses It                    |
| -------- | ---------------------------------------------------- | ---------------------------------------- |
| Person 3 | Trained XGBoost model, slice metrics, Optuna results | Model Health page, live scoring fallback |
| Person 4 | `shap_values.parquet`, bias report                   | SHAP risk drivers on Company Risk page   |
| Person 5 | `scores.parquet`, `manifest.json`, batch inference   | Primary data source for all pages        |
