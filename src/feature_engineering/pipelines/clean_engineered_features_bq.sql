-- =============================================================================
-- Clean Engineered Features Table
-- =============================================================================
-- Creates: ${PROJECT}.${DATASET}.cleaned_engineered_features
-- Source:  ${PROJECT}.${DATASET}.engineered_features
--
-- PURPOSE:
--   The engineered_features table contains significant NULLs (34.9% of all
--   cells across 53 columns) produced by SAFE_DIVIDE returning NULL when
--   the denominator is zero. This SQL creates a cleaned version where every
--   NULL is replaced with a documented, category-appropriate default value.
--
-- ROOT CAUSE OF NULLs:
--   Many SEC EDGAR columns (Revenues, GrossProfit, R&D, SGA, Inventory, etc.)
--   do not exist in the raw_features table and were defaulted to 0.0 during
--   feature engineering. When the SQL computes ratios like GrossProfit/Revenues
--   (0/0), SAFE_DIVIDE correctly returns NULL. These NULLs are NOT missing data
--   in the traditional sense — they mean "this metric is not computable."
--
-- IMPUTATION STRATEGY:
--   All feature NULLs → 0.0 (neutral signal). Detailed reasoning per category:
--
--   FINANCIAL RATIOS (71% null):
--     Denominator was 0 → ratio is undefined → 0.0 = "not applicable"
--
--   GROWTH RATES (87-100% null):
--     No prior-year value or source column is 0 → 0% growth assumed
--
--   ROLLING STATISTICS (27-85% null):
--     Rolling mean: insufficient data points in window → 0.0 (no trend)
--     Rolling std:  ≤1 data point in window → 0.0 (zero volatility)
--
--   COMPOSITE FEATURES (71-100% null):
--     altman_z_approx: components are NULL → 0.0 (score undefined)
--     cash_burn_rate:   all expense columns are 0 → 0.0 (no burn signal)
--     leverage_x_margin: one or both terms NULL → 0.0 (no interaction)
--
--   INTENSITY RATIOS (86% null):
--     rd_intensity, sga_intensity: Revenues=0 → 0.0 (not applicable)
--
--   MACRO INTERACTIONS (62-86% null):
--     macro × NULL_ratio → 0.0 (no interaction signal when ratio undefined)
-- =============================================================================

CREATE OR REPLACE TABLE `${PROJECT}.${DATASET}.cleaned_engineered_features` AS

SELECT
  -- ═══════════════════════════════════════════════════════════════════════
  -- IDENTITY COLUMNS (no NULLs — pass through unchanged)
  -- ═══════════════════════════════════════════════════════════════════════
  firm_id,
  fiscal_year,
  fiscal_period,
  filed_date,
  distress_label,

  -- ═══════════════════════════════════════════════════════════════════════
  -- RAW FINANCIAL COLUMNS (no NULLs — pass through unchanged)
  -- ═══════════════════════════════════════════════════════════════════════
  Assets,
  AssetsCurrent,
  CashAndCashEquivalentsAtCarryingValue,
  Liabilities,
  LiabilitiesCurrent,
  StockholdersEquity,
  RetainedEarningsAccumulatedDeficit,
  NetIncomeLoss,
  OperatingIncomeLoss,
  NetCashProvidedByUsedInOperatingActivities,
  NetCashProvidedByUsedInInvestingActivities,
  NetCashProvidedByUsedInFinancingActivities,
  DepreciationDepletionAndAmortization,
  InterestExpense,
  IncomeTaxExpenseBenefit,
  AccountsReceivableNetCurrent,
  AccountsPayableCurrent,
  PropertyPlantAndEquipmentNet,
  Goodwill,
  CommonStockValue,
  InventoryNet,
  LongTermDebt,
  LongTermDebtCurrent,
  AdditionalPaidInCapital,
  IntangibleAssetsNetExcludingGoodwill,
  Revenues,
  CostOfGoodsAndServicesSold,
  GrossProfit,
  ResearchAndDevelopmentExpense,
  SellingGeneralAndAdministrativeExpense,

  -- ═══════════════════════════════════════════════════════════════════════
  -- MACRO COLUMNS (no NULLs — pass through unchanged)
  -- ═══════════════════════════════════════════════════════════════════════
  fed_funds,
  unemployment,
  inflation,

  -- ═══════════════════════════════════════════════════════════════════════
  -- FINANCIAL RATIOS — ~71% null
  -- Reason: SAFE_DIVIDE(x, 0) = NULL when denominator (LiabilitiesCurrent,
  --         StockholdersEquity, Liabilities, Revenues, Assets) is zero.
  -- Fill: 0.0 — ratio is undefined when denominator = 0, neutral signal.
  -- ═══════════════════════════════════════════════════════════════════════
  IFNULL(current_ratio, 0.0)     AS current_ratio,      -- AssetsCurrent / LiabilitiesCurrent
  IFNULL(quick_ratio, 0.0)       AS quick_ratio,        -- (AssetsCurrent - Inventory) / LiabilitiesCurrent
  IFNULL(cash_ratio, 0.0)        AS cash_ratio,         -- Cash / LiabilitiesCurrent
  IFNULL(debt_to_equity, 0.0)    AS debt_to_equity,     -- Liabilities / StockholdersEquity
  IFNULL(debt_to_assets, 0.0)    AS debt_to_assets,     -- Liabilities / Assets
  IFNULL(interest_coverage, 0.0) AS interest_coverage,  -- OperatingIncome / InterestExpense
  IFNULL(gross_margin, 0.0)      AS gross_margin,       -- GrossProfit / Revenues
  IFNULL(operating_margin, 0.0)  AS operating_margin,   -- OperatingIncome / Revenues
  IFNULL(net_margin, 0.0)        AS net_margin,         -- NetIncome / Revenues
  IFNULL(roa, 0.0)               AS roa,                -- NetIncome / Assets
  IFNULL(roe, 0.0)               AS roe,                -- NetIncome / StockholdersEquity
  IFNULL(asset_turnover, 0.0)    AS asset_turnover,     -- Revenues / Assets
  IFNULL(cash_flow_to_debt, 0.0) AS cash_flow_to_debt,  -- OperatingCF / Liabilities

  -- ═══════════════════════════════════════════════════════════════════════
  -- GROWTH RATES (YoY) — 62-100% null
  -- Reason: LAG(4) returns NULL for first 4 quarters per firm, AND
  --         source columns (Revenues, R&D, SGA, GrossProfit) are 0 →
  --         pct_change(0, 0) is undefined.
  -- Fill: 0.0 — no growth signal available → assume 0% change.
  -- ═══════════════════════════════════════════════════════════════════════
  IFNULL(revenue_growth_yoy, 0.0)      AS revenue_growth_yoy,
  IFNULL(assets_growth_yoy, 0.0)       AS assets_growth_yoy,
  IFNULL(net_income_growth_yoy, 0.0)   AS net_income_growth_yoy,
  IFNULL(operating_cf_growth_yoy, 0.0) AS operating_cf_growth_yoy,
  IFNULL(liabilities_growth_yoy, 0.0)  AS liabilities_growth_yoy,
  IFNULL(rd_growth_yoy, 0.0)           AS rd_growth_yoy,           -- 100% null (R&D col = 0)
  IFNULL(sga_growth_yoy, 0.0)          AS sga_growth_yoy,          -- 100% null (SGA col = 0)
  IFNULL(gross_profit_growth_yoy, 0.0) AS gross_profit_growth_yoy, -- 100% null (GrossProfit col = 0)

  -- ═══════════════════════════════════════════════════════════════════════
  -- ROLLING STATISTICS — 8-85% null
  -- Reason (mean): not enough data points in the rolling window yet
  --   (first N-1 quarters per firm). Fill: 0.0 (no trend data).
  -- Reason (std): ≤1 non-null value in window → STDDEV is undefined.
  --   Fill: 0.0 (zero volatility when insufficient history).
  -- ═══════════════════════════════════════════════════════════════════════
  -- current_ratio rolling
  IFNULL(current_ratio_rolling_4q_mean, 0.0) AS current_ratio_rolling_4q_mean,
  IFNULL(current_ratio_rolling_4q_std, 0.0)  AS current_ratio_rolling_4q_std,
  IFNULL(current_ratio_rolling_8q_mean, 0.0) AS current_ratio_rolling_8q_mean,
  IFNULL(current_ratio_rolling_8q_std, 0.0)  AS current_ratio_rolling_8q_std,
  -- debt_to_equity rolling
  IFNULL(debt_to_equity_rolling_4q_mean, 0.0) AS debt_to_equity_rolling_4q_mean,
  IFNULL(debt_to_equity_rolling_4q_std, 0.0)  AS debt_to_equity_rolling_4q_std,
  IFNULL(debt_to_equity_rolling_8q_mean, 0.0) AS debt_to_equity_rolling_8q_mean,
  IFNULL(debt_to_equity_rolling_8q_std, 0.0)  AS debt_to_equity_rolling_8q_std,
  -- net_margin rolling
  IFNULL(net_margin_rolling_4q_mean, 0.0) AS net_margin_rolling_4q_mean,
  IFNULL(net_margin_rolling_4q_std, 0.0)  AS net_margin_rolling_4q_std,
  IFNULL(net_margin_rolling_8q_mean, 0.0) AS net_margin_rolling_8q_mean,
  IFNULL(net_margin_rolling_8q_std, 0.0)  AS net_margin_rolling_8q_std,
  -- roa rolling
  IFNULL(roa_rolling_4q_mean, 0.0) AS roa_rolling_4q_mean,
  IFNULL(roa_rolling_4q_std, 0.0)  AS roa_rolling_4q_std,
  IFNULL(roa_rolling_8q_mean, 0.0) AS roa_rolling_8q_mean,
  IFNULL(roa_rolling_8q_std, 0.0)  AS roa_rolling_8q_std,
  -- cash_flow_to_debt rolling
  IFNULL(cash_flow_to_debt_rolling_4q_mean, 0.0) AS cash_flow_to_debt_rolling_4q_mean,
  IFNULL(cash_flow_to_debt_rolling_4q_std, 0.0)  AS cash_flow_to_debt_rolling_4q_std,
  IFNULL(cash_flow_to_debt_rolling_8q_mean, 0.0) AS cash_flow_to_debt_rolling_8q_mean,
  IFNULL(cash_flow_to_debt_rolling_8q_std, 0.0)  AS cash_flow_to_debt_rolling_8q_std,
  -- revenue_growth_yoy rolling
  IFNULL(revenue_growth_yoy_rolling_4q_mean, 0.0) AS revenue_growth_yoy_rolling_4q_mean,
  IFNULL(revenue_growth_yoy_rolling_4q_std, 0.0)  AS revenue_growth_yoy_rolling_4q_std,
  IFNULL(revenue_growth_yoy_rolling_8q_mean, 0.0) AS revenue_growth_yoy_rolling_8q_mean,
  IFNULL(revenue_growth_yoy_rolling_8q_std, 0.0)  AS revenue_growth_yoy_rolling_8q_std,

  -- ═══════════════════════════════════════════════════════════════════════
  -- COMPOSITE & INTERACTION FEATURES — 71-100% null
  -- Reason: component ratios are NULL → NULL propagates through formulas.
  -- Fill: 0.0 — composite is undefined when components are undefined.
  -- ═══════════════════════════════════════════════════════════════════════
  IFNULL(altman_z_approx, 0.0)   AS altman_z_approx,    -- 1.2*X1 + 1.4*X2 + ... (NULL components)
  IFNULL(cash_burn_rate, 0.0)    AS cash_burn_rate,      -- 100% null: all expense cols = 0
  IFNULL(leverage_x_margin, 0.0) AS leverage_x_margin,   -- debt_to_equity * operating_margin
  IFNULL(rd_intensity, 0.0)      AS rd_intensity,        -- R&D / Revenues (both zero)
  IFNULL(sga_intensity, 0.0)     AS sga_intensity,       -- SGA / Revenues (both zero)

  -- ═══════════════════════════════════════════════════════════════════════
  -- MACRO INTERACTION FEATURES — 62-86% null
  -- Reason: macro_value × NULL_ratio → NULL. The macro values themselves
  --         are non-null, but the financial ratio they interact with is NULL.
  -- Fill: 0.0 — no interaction when the ratio component is undefined.
  -- ═══════════════════════════════════════════════════════════════════════
  IFNULL(fed_rate_x_leverage, 0.0)    AS fed_rate_x_leverage,    -- fed_funds × debt_to_equity
  IFNULL(unemployment_x_margin, 0.0)  AS unemployment_x_margin,  -- unemployment × net_margin
  IFNULL(inflation_x_cash_ratio, 0.0) AS inflation_x_cash_ratio, -- inflation × cash_ratio

  -- ═══════════════════════════════════════════════════════════════════════
  -- CATEGORICAL COLUMNS (no NULLs — pass through unchanged)
  -- ═══════════════════════════════════════════════════════════════════════
  company_size_bucket,
  sector_proxy

FROM `${PROJECT}.${DATASET}.engineered_features`;
