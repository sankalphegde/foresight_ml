# Foresight-ML: Corporate Financial Distress Early-Warning System

**Foresight-ML** is an end-to-end MLOps initiative designed to predict corporate financial distress before it becomes irreversible. By leveraging historical financial data and machine learning, this system offers a dynamic alternative to static, lagging financial indicators.

---

## 1. Project Description

The current landscape of corporate financial health monitoring suffers from inefficiencies that lead to "surprise" bankruptcies and delayed interventions. This project addresses two core problems:

* **Latency in Detection:** Financial distress is typically identified only after official quarterly reports (10-Q/10-K) are released. By the time a report is analyzed, the company may have been in distress for months.
* **Static & Outdated Thresholds:** Traditional methods rely on rigid rules (e.g., "Debt-to-Equity > 2.0"). These fail to adapt to changing macroeconomic conditions, such as shifting interest rate environments or industry-specific nuances.

**Foresight-ML** solves this by treating financial distress as a time-series classification problem, updating risk scores in near real-time as new market data becomes available.

---

## 2. Dataset Sources

This project utilizes a combination of fundamental and market data. The raw data is versioned and managed using **DVC (Data Version Control)** to ensure reproducibility.
All data used in this project is **publicly available**, ensuring transparency, reproducibility, and suitability for academic research.

### Primary Data Source
- **SEC EDGAR**  
  - 10-K (annual) and 10-Q (quarterly) filings  
  - Structured XBRL financial statements  
  - Income statements, balance sheets, cash flow statements, and selected disclosures  

### Supplementary Data Sources
- **Federal Reserve Economic Data (FRED)**  
  - Federal Funds Rate  
  - Inflation (CPI)  
  - Credit spreads  
  - Unemployment and growth indicators  

- **Public Distress Labels**
  - Bankruptcy filings (e.g., UCLA LoPucki Bankruptcy Database)
  - Exchange delisting records due to financial non-compliance
  - Publicly available financial distress datasets (e.g., Kaggle)

### Optional Enhancements
- **Market Data**
  - Stock prices, returns, volatility, and trading volume
  - Sourced via Yahoo Finance (`yfinance`) or Alpha Vantage

* **Data Management:**
    * Raw data is stored in remote object storage (GCP bucket).
    * `dvc.yaml` defines the data pipeline stages (ingest, clean, split).
    * To access the data locally, you must have GCP credentials configured and run `dvc pull`.

---

## 3. Dataset Description

### Data Schema & Shape
The dataset is structured at the **company–quarter level**, where each row represents a single firm’s financial state for a given fiscal quarter.

**Expected dataset characteristics (initial):**
- Rows: ~80,000 company-quarter observations  
- Columns: 40–80 engineered features + metadata  
- Time span: Multiple years (2010–2023)

This structure supports longitudinal analysis and time-based modeling.

---

### Key Feature Groups
Rather than relying on raw financial line items alone, the project focuses on **engineered financial indicators** commonly used in corporate finance and risk analysis:

#### Liquidity Metrics
- Current Ratio  
- Quick Ratio  

#### Leverage & Solvency Metrics
- Debt-to-Equity  
- Total Debt / Total Assets  
- Interest Coverage Ratio  

#### Profitability Metrics
- Net Margin  
- Return on Assets (ROA)  
- Operating Margin  

#### Cash Flow Indicators
- Operating Cash Flow / Total Debt  
- Free Cash Flow trends  

#### Temporal Features
- Quarter-over-quarter growth rates (revenue, cash flow)
- Rolling 4-quarter slopes and volatility measures

#### Macroeconomic Indicators
- Interest rates
- Inflation
- Credit spreads

Optional NLP-derived features may be added from textual disclosures but are treated as non-blocking enhancements.

---

### Missing & Null Values
Missing values are expected due to:
- Incomplete filings or reporting gaps
- Newly listed companies with limited history
- Undefined ratios caused by zero or near-zero denominators

Exact missing-value counts will be quantified during exploratory data analysis (EDA).

---

### Data Quality Considerations
Financial datasets often contain inconsistencies and anomalies, including:
- Outliers caused by accounting restatements
- Inconsistent XBRL tagging across firms
- Duplicate filings for the same fiscal period

To address these issues, the project includes:
- Accounting identity checks (Assets = Liabilities + Equity)
- Outlier handling via winsorization
- Deduplication by fiscal period
- Unit tests for feature calculations

---

## 4. Target Variable
The target variable is a binary classification label:

**`DistressNext12Months`**
- `1`: The company experiences a publicly observable distress event within the next 12 months  
- `0`: No distress event observed within the next 12 months  

Distress events are defined conservatively using **objective, verifiable public outcomes**, such as bankruptcy filings or financial delistings, to avoid reliance on proprietary credit rating data.

---

## 5. Planned Data Processing & Splits
The data pipeline is designed to prevent information leakage and reflect real-world deployment constraints.

Key preprocessing steps include:
- Time-based train/validation/test splits
- Forward-fill imputation for short gaps
- Sector-median fallback for longer gaps
- Outlier handling via winsorization
- Feature scaling computed on training data only

### Data Splitting Strategy
- **Training:** 2010–2019  
- **Validation:** 2020–2021 (stress-tested on COVID period)  
- **Test:** 2022–2023  

This approach ensures realistic evaluation under changing economic regimes.

---

## 6. Model Output
The system outputs:
- A **probability score between 0 and 1** representing the estimated likelihood of financial distress within the next 12 months
- **Feature-level explanations** (e.g., SHAP values) highlighting the main drivers of risk

These outputs are designed to be interpretable and actionable for non-technical users.

---

## 7. Research & Prior Work
This project is informed by established financial risk research, including:
- Altman Z-score and other ratio-based bankruptcy models
- Credit ratings as industry benchmarks (not used directly due to access constraints)
- Prior evidence that machine learning models incorporating temporal and macroeconomic features can improve early detection of distress

The project builds upon this work by emphasizing automation, scalability, and explainability.

---

## 8. Project Status & Next Steps
**Current Status:**
- Data ingestion pipeline setup
- Schema validation and initial EDA

**Next Steps:**
- Feature engineering and baseline model training
- Model evaluation and explainability analysis
- Deployment of batch inference and on-demand scoring API
- Dashboard development for visualization and demo

---
## 9. Setup Instructions

Follow these steps to set up the development environment for local experimentation and pipeline execution.

### Prerequisites
- Python 3.9+
- Docker (for containerized services and jobs)
- Git
- DVC (for data versioning)

> Note: Cloud resources are managed using Google Cloud Platform (GCP) services such as Cloud Run, Cloud Scheduler, and Cloud Storage. No Terraform or Kubernetes setup is required.

---

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/sankalphegde/Foresight-ML.git
cd Foresight-ML
```

2. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```

3. **Initialize Code Quality Hooks**
This project uses pre-commit to enforce code standards.
```bash
pre-commit install
```

4. **Pull Versioned Data**
If DVC remotes are configured, pull the latest datasets and artifacts:
```bash
dvc pull
```

If data is not yet available via DVC, ingestion scripts can be run to fetch raw data from public sources.

5. **Cloud Deployment**

Deployment to Google Cloud Platform is handled using Cloud Run services and jobs, triggered via Cloud Scheduler and GitHub Actions.
Deployment configuration files are located in the infra/ directory.

This step is optional for local development and required only for cloud execution or demo deployment.

## 10. Example Usage

You can interact with the system either by running pipeline stages locally or by invoking deployed cloud services.

### Running the Training Pipeline

To execute the full batch pipeline (data preprocessing → feature engineering → model training):
```bash
dvc repro
```

This command reproduces the pipeline defined in dvc.yaml, ensuring reproducibility across runs.

### Running the API Locally (Optional)

To start the FastAPI service locally for testing:
```bash
uvicorn src.api.main:app --reload
```

The API will be available at:
```bash
http://localhost:8000
```

### On-Demand Inference (Demo Mode)
When deployed to Cloud Run, the API supports on-demand scoring for selected companies, returning:
- Distress probability
- Key contributing features
