# Financial Distress Pipeline — Bias Analysis Report

*Generated: 2026-02-21 20:23:02*

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Total Observations | 211,614 |
| Unique Firms | 3,847 |
| Year Range | 2009–2026 |
| Engineered Features | 20 |
| Bias Dimensions | 5 |
| High-Drift Alerts | 3 |
| Distress Rate | 1.76% (3,734 / 211,614) |
| Class Imbalance Ratio | 1:56 |

> [!WARNING]
> **3 high-drift alerts detected.** Features exhibiting PSI > 0.25 across slices
> need investigation before model training.

## 2. Missing Data Analysis

### 2.1 Raw Features Table (`raw_features`)

The source table is **remarkably clean** with <0.01% overall null rate:

| Category | Columns | Null Rate | Handling Strategy |
|---|---|---|---|
| Identity (firm_id, fiscal_year, etc.) | 4 | 0.0% | No action needed |
| Financial statements (Assets, Liabilities, etc.) | 28 | 0.0% | No action needed |
| Macroeconomic (FedFundsRate, CPI, etc.) | 6 | <0.01% (20 rows) | Forward-fill by date, then backfill |
| Lag columns (total_assets_lag1, etc.) | 4 | 1.8–7.2% | Fill with current-period value (0% change) |
| Distress label | 1 | 0.0% | No action needed |

### 2.2 Engineered Features Table (Post-SQL)

The feature engineering SQL produces significant NULLs due to `SAFE_DIVIDE` by zero:

| Feature Category | Null % | Root Cause | Imputation |
|---|---|---|---|
| Financial ratios | ~71% | SAFE_DIVIDE(x, 0) = NULL | Fill 0.0 — ratio undefined when denom=0 |
| Growth rates (YoY) | ~87% | LAG(4) + SAFE_DIVIDE by 0 | Fill 0.0 — no growth signal available |
| Rolling std | ~65% | STDDEV over ≤1 non-null value | Fill 0.0 — zero volatility |
| Altman Z-score | ~71% | Compound NULL propagation | Fill 0.0 — composite undefined |
| Cash burn rate | 100% | All expense columns = 0 | Fill 0.0 — no burn rate signal |

**Key Insight**: The NULLs are *not* missing data in the traditional sense. They arise because
many SEC EDGAR columns expected by the SQL (Revenues, GrossProfit, R&D, SGA, Inventory, etc.)
were not present in the raw BigQuery table and defaulted to 0.0. When the SQL computes ratios
like `Revenues / Assets`, the result is a valid 0.0. But when it computes `GrossProfit / Revenues`
(0 / 0), SAFE_DIVIDE correctly returns NULL. Filling with 0.0 is appropriate because it
represents 'this metric is not applicable for this observation.'

## 3. Label Imbalance Analysis

The dataset has a **severe class imbalance** with only 1.76% positive labels 
(3,734 distressed vs 207,880 non-distressed).

> [!CAUTION]
> With a 1:56 imbalance ratio, standard classifiers will be biased
> toward predicting the majority class. **Mitigation strategies**: SMOTE oversampling,
> class-weighted loss functions, or focal loss. Evaluate with PR-AUC, not accuracy.

### 3.1 Distress Rate by Slice

**Company Size**:

| Slice | Count | Distress Rate |
|---|---|---|
| large | 52,903 | 3.49% |
| mega | 52,903 | 1.06% |
| mid | 52,904 | 1.77% |
| small | 52,904 | 0.74% |

**Sector Proxy**:

| Slice | Count | Distress Rate |
|---|---|---|
| financial_capital_intensive | 5,893 | 0.14% |
| services_other | 205,721 | 1.81% |

### 3.2 Feature Separation by Label (Healthy vs Distressed)

This section measures how each feature's distribution differs between
healthy (label=0) and distressed (label=1) firms. Features with large
separation are strong predictive signals — but extreme values may also
indicate data leakage if they are outcomes of distress rather than causes.

#### 3.2a Feature Distribution Comparison

| Feature | Mean (Healthy) | Mean (Distressed) | Cohen's d | KS Stat | KS p-value | PSI | Signal |
|---|---|---|---|---|---|---|---|
| current_ratio | 0.6514 | 2.2363 | +0.101 | 0.117 | 1.74e-44 | 0.002 | ⚪ NEGLIGIBLE |
| quick_ratio | 0.6514 | 2.2363 | +0.101 | 0.117 | 1.74e-44 | 0.002 | ⚪ NEGLIGIBLE |
| cash_ratio | 0.2405 | 1.0709 | +0.085 | 0.128 | 1.35e-52 | 0.005 | ⚪ NEGLIGIBLE |
| debt_to_equity | 3.5506 | 0.4119 | -0.003 | 0.108 | 1.87e-37 | 0.000 | ⚪ NEGLIGIBLE |
| debt_to_assets | 0.7935 | 8.2991 | +0.089 | 0.141 | 5.61e-64 | 0.005 | ⚪ NEGLIGIBLE |
| interest_coverage | -89.6196 | -30.7541 | +0.002 | 0.095 | 1.51e-29 | 0.000 | ⚪ NEGLIGIBLE |
| gross_margin | 0.0000 | 0.0000 | +0.000 | 0.000 | 1.00e+00 | 0.000 | ⚪ NEGLIGIBLE |
| operating_margin | 0.0372 | 0.1072 | +0.004 | 0.054 | 1.17e-09 | 0.000 | ⚪ NEGLIGIBLE |
| net_margin | 0.2856 | 0.2071 | -0.001 | 0.156 | 1.22e-78 | 0.000 | ⚪ NEGLIGIBLE |
| roa | -0.1297 | -8.1338 | -0.126 | 0.520 | 0.00e+00 | 0.005 | ⚪ NEGLIGIBLE |
| roe | -0.0099 | -0.0123 | -0.000 | 0.418 | 0.00e+00 | 0.000 | ⚪ NEGLIGIBLE |
| asset_turnover | -0.0248 | -0.0629 | -0.006 | 0.114 | 3.22e-42 | 0.000 | ⚪ NEGLIGIBLE |
| cash_flow_to_debt | -0.0430 | -0.5620 | -0.024 | 0.166 | 5.64e-89 | 0.002 | ⚪ NEGLIGIBLE |
| revenue_growth_yoy | 0.0987 | -0.0781 | -0.001 | 0.031 | 1.52e-03 | 0.000 | ⚪ NEGLIGIBLE |
| net_income_growth_yoy | 15.6129 | -1.6850 | -0.002 | 0.112 | 8.53e-41 | 0.000 | ⚪ NEGLIGIBLE |
| altman_z_approx | -1.9408 | -124.8505 | -0.090 | 0.120 | 2.16e-46 | 0.007 | ⚪ NEGLIGIBLE |
| cash_burn_rate | 0.0000 | 0.0000 | +0.000 | 0.000 | 1.00e+00 | 0.000 | ⚪ NEGLIGIBLE |
| rd_intensity | 0.0000 | 0.0000 | +0.000 | 0.000 | 1.00e+00 | 0.000 | ⚪ NEGLIGIBLE |
| sga_intensity | 0.0000 | 0.0000 | +0.000 | 0.000 | 1.00e+00 | 0.000 | ⚪ NEGLIGIBLE |
| leverage_x_margin | 1.7395 | 0.1787 | -0.002 | 0.032 | 1.06e-03 | 0.000 | ⚪ NEGLIGIBLE |

> [!NOTE]
> **Effect size summary**: 0 large / 0 medium / 0 small / 20 negligible out of 20 features.
> Cohen's d thresholds: |d| ≥ 0.8 large, ≥ 0.5 medium, ≥ 0.2 small.

#### 3.2b Chi-Squared Independence Tests

Tests whether `distress_label` is statistically independent of each
categorical slice. A low p-value means distress is unevenly distributed.

| Dimension | Chi² Statistic | p-value | Degrees of Freedom | Independent? |
|---|---|---|---|---|
| Company Size | 1,381.0 | 3.86e-299 | 3 | ❌ No |
| Sector Proxy | 91.8 | 9.54e-22 | 1 | ❌ No |
| Time Period | 68.6 | 1.19e-16 | 1 | ❌ No |

#### 3.2c Disparate Impact Analysis

Disparate Impact Ratio (DIR) = distress_rate(group) / distress_rate(reference_group).
A DIR < 0.8 or > 1.25 indicates potential unfair treatment per the 80% rule.

**Company Size** (reference: large at 3.49%):

| Group | Distress Rate | DIR | Fair? |
|---|---|---|---|
| large | 3.49% | 1.000 | ✅ |
| mega | 1.06% | 0.305 | ⚠️ UNFAIR |
| mid | 1.77% | 0.507 | ⚠️ UNFAIR |
| small | 0.74% | 0.211 | ⚠️ UNFAIR |

**Sector Proxy** (reference: services_other at 1.81%):

| Group | Distress Rate | DIR | Fair? |
|---|---|---|---|
| financial_capital_intensive | 0.14% | 0.075 | ⚠️ UNFAIR |
| services_other | 1.81% | 1.000 | ✅ |

## 4. Feature Drift Detection (PSI)

Population Stability Index (PSI) measures distributional shift between slices:

| PSI Range | Interpretation |
|---|---|
| < 0.10 | No significant shift — feature is stable |
| 0.10–0.25 | Moderate shift — investigate |
| > 0.25 | Significant shift — action required |

### 4.1 High-Drift Alerts

- ⚠ HIGH DRIFT: debt_to_assets between mega and mid (PSI=15.172)
- ⚠ HIGH DRIFT: roa between mega and mid (PSI=2.705)
- ⚠ HIGH DRIFT: asset_turnover between mega and mid (PSI=0.346)

### 4.2 Drift Summary by Dimension

**Company Size**: 3 high / 0 moderate / 117 stable features
**Sector Proxy**: 0 high / 0 moderate / 20 stable features
**Time Period**: 0 high / 0 moderate / 20 stable features
**Macro Regime**: 0 high / 0 moderate / 20 stable features
**Distress Label**: 0 high / 0 moderate / 20 stable features

## 5. Slice Statistics Summary

### Company Size

| Slice | Samples | % of Total |
|---|---|---|
| large | 52,903 | 25.0% |
| mega | 52,903 | 25.0% |
| mid | 52,904 | 25.0% |
| small | 52,904 | 25.0% |

### Sector Proxy

| Slice | Samples | % of Total |
|---|---|---|
| financial_capital_intensive | 5,893 | 2.8% |
| services_other | 205,721 | 97.2% |

### Time Period

| Slice | Samples | % of Total |
|---|---|---|
| pre_2016 | 59,571 | 28.2% |
| post_2016 | 152,043 | 71.8% |

### Macro Regime

| Slice | Samples | % of Total |
|---|---|---|
| low_rate | 138,903 | 65.6% |
| high_rate | 72,711 | 34.4% |

### Distress Label

| Slice | Samples | % of Total |
|---|---|---|
| healthy (label=0) | 207,880 | 98.2% |
| distressed (label=1) | 3,734 | 1.8% |

## 6. Recommendations

1. **Address class imbalance** before training: Use SMOTE, class weights, or stratified 
   sampling. Evaluate with PR-AUC and F1, not accuracy.
2. **Investigate high-drift features**: Features with PSI > 0.25 across company size 
   buckets may cause the model to perform differently for small vs. mega-cap firms.
3. **Consider removing zero-filled features**: Features that are 100% NULL (like 
   `cash_burn_rate`) have no discriminative power after 0-fill and should be dropped 
   or sourced from alternative data.
4. **Stratified evaluation**: Always evaluate model performance per-slice (size, sector, 
   time period) to detect hidden bias in predictions.
5. **Temporal validation**: Use a walk-forward split (not random) to prevent lookahead bias, 
   given the 2009–2026 time span and identified temporal drift.
