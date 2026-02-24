-- ============================================================
-- Feature Engineering Pipeline — BigQuery SQL
-- ============================================================
-- Mirrors the Python feature_engineering.py logic exactly.
-- Designed for scalable execution on BigQuery.
--
-- Usage:
--   1. Replace ${PROJECT}, ${DATASET}, ${RAW_TABLE} with your values
--   2. Run: bq query --use_legacy_sql=false < feature_engineering_bq.sql
--   3. Or execute via Python BigQuery client (see run_pipeline.py)
--
-- This script:
--   a) Cleans data (drops columns, imputes NaN)
--   b) Computes all engineered features
--   c) Writes to ${DATASET}.engineered_features
-- ============================================================

CREATE OR REPLACE TABLE `${PROJECT}.${DATASET}.engineered_features` AS

WITH

-- ─────────────────────────────────────────────────────────────
-- Step 1: Data Cleaning
-- Map actual column names to canonical names used in feature engineering
-- Forward-fill macro columns by date
-- ─────────────────────────────────────────────────────────────
cleaned AS (
  SELECT
    firm_id,
    fiscal_year,
    fiscal_period,
    filed_date,

    -- Core financials (mapped from actual column names)
    IFNULL(total_assets, 0)                                          AS Assets,
    IFNULL(AssetsCurrent, 0)                                         AS AssetsCurrent,
    IFNULL(CashAndCashEquivalentsAtCarryingValue, 0)                 AS CashAndCashEquivalentsAtCarryingValue,
    IFNULL(total_liabilities, 0)                                     AS Liabilities,
    IFNULL(LiabilitiesCurrent, 0)                                    AS LiabilitiesCurrent,
    IFNULL(StockholdersEquity, 0)                                    AS StockholdersEquity,
    IFNULL(RetainedEarningsAccumulatedDeficit, 0)                    AS RetainedEarningsAccumulatedDeficit,
    IFNULL(net_income, 0)                                            AS NetIncomeLoss,
    IFNULL(OperatingIncomeLoss, 0)                                   AS OperatingIncomeLoss,
    IFNULL(NetCashProvidedByUsedInOperatingActivities, 0)            AS NetCashProvidedByUsedInOperatingActivities,
    IFNULL(NetCashProvidedByUsedInInvestingActivities, 0)            AS NetCashProvidedByUsedInInvestingActivities,
    IFNULL(NetCashProvidedByUsedInFinancingActivities, 0)            AS NetCashProvidedByUsedInFinancingActivities,
    IFNULL(DepreciationDepletionAndAmortization, 0)                  AS DepreciationDepletionAndAmortization,
    IFNULL(InterestExpense, 0)                                       AS InterestExpense,
    IFNULL(IncomeTaxExpenseBenefit, 0)                               AS IncomeTaxExpenseBenefit,
    IFNULL(AccountsReceivableNetCurrent, 0)                          AS AccountsReceivableNetCurrent,
    IFNULL(AccountsPayableCurrent, 0)                                AS AccountsPayableCurrent,
    IFNULL(PropertyPlantAndEquipmentNet, 0)                          AS PropertyPlantAndEquipmentNet,
    IFNULL(Goodwill, 0)                                              AS Goodwill,
    IFNULL(CommonStockValue, 0)                                      AS CommonStockValue,

    -- Columns not present in raw_features — default to 0
    0.0 AS InventoryNet,
    0.0 AS LongTermDebt,
    0.0 AS LongTermDebtCurrent,
    0.0 AS AdditionalPaidInCapital,
    0.0 AS IntangibleAssetsNetExcludingGoodwill,

    -- Revenue proxy: use OperatingIncomeLoss + net_income as fallback
    -- (table has no Revenues column; use ProfitLoss if available, else 0)
    IFNULL(ProfitLoss, 0)                                            AS Revenues,

    -- Cost/expense proxies (not in raw table — default to 0)
    0.0 AS CostOfGoodsAndServicesSold,
    0.0 AS GrossProfit,
    0.0 AS ResearchAndDevelopmentExpense,
    0.0 AS SellingGeneralAndAdministrativeExpense,

    -- Distress label (pass through)
    distress_label,

    -- Macro columns: forward-fill by date, then backfill
    COALESCE(
      FedFundsRate,
      LAST_VALUE(FedFundsRate IGNORE NULLS) OVER (ORDER BY filed_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
      FIRST_VALUE(FedFundsRate IGNORE NULLS) OVER (ORDER BY filed_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
    ) AS fed_funds,

    COALESCE(
      UnemploymentRate,
      LAST_VALUE(UnemploymentRate IGNORE NULLS) OVER (ORDER BY filed_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
      FIRST_VALUE(UnemploymentRate IGNORE NULLS) OVER (ORDER BY filed_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
    ) AS unemployment,

    COALESCE(
      CPI,
      LAST_VALUE(CPI IGNORE NULLS) OVER (ORDER BY filed_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
      FIRST_VALUE(CPI IGNORE NULLS) OVER (ORDER BY filed_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
    ) AS inflation

  FROM `${PROJECT}.${DATASET}.${RAW_TABLE}`
),

-- ─────────────────────────────────────────────────────────────
-- Step 2: Row numbering for window functions
-- ─────────────────────────────────────────────────────────────
ordered AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY firm_id
      ORDER BY fiscal_year,
        CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END
    ) AS row_num
  FROM cleaned
),

-- ─────────────────────────────────────────────────────────────
-- Step 3: Financial Ratios (13 features)
-- ─────────────────────────────────────────────────────────────
with_ratios AS (
  SELECT
    *,

    -- Liquidity
    SAFE_DIVIDE(AssetsCurrent, LiabilitiesCurrent) AS current_ratio,
    SAFE_DIVIDE(AssetsCurrent - InventoryNet, LiabilitiesCurrent) AS quick_ratio,
    SAFE_DIVIDE(CashAndCashEquivalentsAtCarryingValue, LiabilitiesCurrent) AS cash_ratio,

    -- Leverage
    SAFE_DIVIDE(Liabilities, StockholdersEquity) AS debt_to_equity,
    SAFE_DIVIDE(Liabilities, Assets) AS debt_to_assets,
    SAFE_DIVIDE(OperatingIncomeLoss, InterestExpense) AS interest_coverage,

    -- Profitability
    SAFE_DIVIDE(GrossProfit, Revenues) AS gross_margin,
    SAFE_DIVIDE(OperatingIncomeLoss, Revenues) AS operating_margin,
    SAFE_DIVIDE(NetIncomeLoss, Revenues) AS net_margin,
    SAFE_DIVIDE(NetIncomeLoss, Assets) AS roa,
    SAFE_DIVIDE(NetIncomeLoss, StockholdersEquity) AS roe,

    -- Efficiency
    SAFE_DIVIDE(Revenues, Assets) AS asset_turnover,

    -- Cash Flow
    SAFE_DIVIDE(NetCashProvidedByUsedInOperatingActivities, Liabilities) AS cash_flow_to_debt

  FROM ordered
),

-- ─────────────────────────────────────────────────────────────
-- Step 4: Growth Rates (YoY = lag 4 quarters)
-- ─────────────────────────────────────────────────────────────
with_growth AS (
  SELECT
    *,

    SAFE_DIVIDE(
      Revenues - LAG(Revenues, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END),
      ABS(LAG(Revenues, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END))
    ) AS revenue_growth_yoy,

    SAFE_DIVIDE(
      Assets - LAG(Assets, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END),
      ABS(LAG(Assets, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END))
    ) AS assets_growth_yoy,

    SAFE_DIVIDE(
      NetIncomeLoss - LAG(NetIncomeLoss, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END),
      ABS(LAG(NetIncomeLoss, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END))
    ) AS net_income_growth_yoy,

    SAFE_DIVIDE(
      NetCashProvidedByUsedInOperatingActivities - LAG(NetCashProvidedByUsedInOperatingActivities, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END),
      ABS(LAG(NetCashProvidedByUsedInOperatingActivities, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END))
    ) AS operating_cf_growth_yoy,

    SAFE_DIVIDE(
      Liabilities - LAG(Liabilities, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END),
      ABS(LAG(Liabilities, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END))
    ) AS liabilities_growth_yoy,

    SAFE_DIVIDE(
      ResearchAndDevelopmentExpense - LAG(ResearchAndDevelopmentExpense, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END),
      ABS(LAG(ResearchAndDevelopmentExpense, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END))
    ) AS rd_growth_yoy,

    SAFE_DIVIDE(
      SellingGeneralAndAdministrativeExpense - LAG(SellingGeneralAndAdministrativeExpense, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END),
      ABS(LAG(SellingGeneralAndAdministrativeExpense, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END))
    ) AS sga_growth_yoy,

    SAFE_DIVIDE(
      GrossProfit - LAG(GrossProfit, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END),
      ABS(LAG(GrossProfit, 4) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END))
    ) AS gross_profit_growth_yoy

  FROM with_ratios
),

-- ─────────────────────────────────────────────────────────────
-- Step 5: Rolling Statistics (4Q and 8Q mean + std)
-- ─────────────────────────────────────────────────────────────
with_rolling AS (
  SELECT
    *,

    -- current_ratio rolling
    AVG(current_ratio) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS current_ratio_rolling_4q_mean,
    STDDEV(current_ratio) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS current_ratio_rolling_4q_std,
    AVG(current_ratio) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS current_ratio_rolling_8q_mean,
    STDDEV(current_ratio) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS current_ratio_rolling_8q_std,

    -- debt_to_equity rolling
    AVG(debt_to_equity) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS debt_to_equity_rolling_4q_mean,
    STDDEV(debt_to_equity) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS debt_to_equity_rolling_4q_std,
    AVG(debt_to_equity) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS debt_to_equity_rolling_8q_mean,
    STDDEV(debt_to_equity) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS debt_to_equity_rolling_8q_std,

    -- net_margin rolling
    AVG(net_margin) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS net_margin_rolling_4q_mean,
    STDDEV(net_margin) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS net_margin_rolling_4q_std,
    AVG(net_margin) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS net_margin_rolling_8q_mean,
    STDDEV(net_margin) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS net_margin_rolling_8q_std,

    -- roa rolling
    AVG(roa) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS roa_rolling_4q_mean,
    STDDEV(roa) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS roa_rolling_4q_std,
    AVG(roa) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS roa_rolling_8q_mean,
    STDDEV(roa) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS roa_rolling_8q_std,

    -- cash_flow_to_debt rolling
    AVG(cash_flow_to_debt) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS cash_flow_to_debt_rolling_4q_mean,
    STDDEV(cash_flow_to_debt) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS cash_flow_to_debt_rolling_4q_std,
    AVG(cash_flow_to_debt) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS cash_flow_to_debt_rolling_8q_mean,
    STDDEV(cash_flow_to_debt) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS cash_flow_to_debt_rolling_8q_std,

    -- revenue_growth_yoy rolling
    AVG(revenue_growth_yoy) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS revenue_growth_yoy_rolling_4q_mean,
    STDDEV(revenue_growth_yoy) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS revenue_growth_yoy_rolling_4q_std,
    AVG(revenue_growth_yoy) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS revenue_growth_yoy_rolling_8q_mean,
    STDDEV(revenue_growth_yoy) OVER (PARTITION BY firm_id ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS revenue_growth_yoy_rolling_8q_std

  FROM with_growth
),

-- ─────────────────────────────────────────────────────────────
-- Step 6: Z-Score, Interaction & Macro Features
-- ─────────────────────────────────────────────────────────────
with_interactions AS (
  SELECT
    *,

    -- Altman Z-score approximation
    1.2 * SAFE_DIVIDE(AssetsCurrent - LiabilitiesCurrent, Assets)
    + 1.4 * SAFE_DIVIDE(RetainedEarningsAccumulatedDeficit, Assets)
    + 3.3 * SAFE_DIVIDE(OperatingIncomeLoss, Assets)
    + 0.6 * SAFE_DIVIDE(StockholdersEquity, Liabilities)
    + 1.0 * SAFE_DIVIDE(Revenues, Assets)
    AS altman_z_approx,

    -- Cash burn rate
    SAFE_DIVIDE(
      CashAndCashEquivalentsAtCarryingValue
        - LAG(CashAndCashEquivalentsAtCarryingValue, 1) OVER (
            PARTITION BY firm_id
            ORDER BY fiscal_year, CASE fiscal_period WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END
          ),
      CostOfGoodsAndServicesSold + SellingGeneralAndAdministrativeExpense + ResearchAndDevelopmentExpense
    ) AS cash_burn_rate,

    -- Leverage × Margin interaction
    SAFE_DIVIDE(Liabilities, StockholdersEquity) * SAFE_DIVIDE(OperatingIncomeLoss, Revenues) AS leverage_x_margin,

    -- R&D intensity
    SAFE_DIVIDE(ResearchAndDevelopmentExpense, Revenues) AS rd_intensity,

    -- SGA intensity
    SAFE_DIVIDE(SellingGeneralAndAdministrativeExpense, Revenues) AS sga_intensity,

    -- Macro interaction features
    fed_funds * SAFE_DIVIDE(Liabilities, StockholdersEquity) AS fed_rate_x_leverage,
    unemployment * SAFE_DIVIDE(NetIncomeLoss, Revenues) AS unemployment_x_margin,
    inflation * SAFE_DIVIDE(CashAndCashEquivalentsAtCarryingValue, LiabilitiesCurrent) AS inflation_x_cash_ratio,

    -- Company size bucket (quartile)
    CASE NTILE(4) OVER (ORDER BY Assets)
      WHEN 1 THEN 'small'
      WHEN 2 THEN 'mid'
      WHEN 3 THEN 'large'
      WHEN 4 THEN 'mega'
    END AS company_size_bucket,

    -- Sector proxy
    CASE
      WHEN SAFE_DIVIDE(ResearchAndDevelopmentExpense, Revenues) > 0.15 THEN 'tech_pharma'
      WHEN SAFE_DIVIDE(InventoryNet, Assets) > 0.20 THEN 'manufacturing_retail'
      WHEN SAFE_DIVIDE(InterestExpense, Revenues) > 0.05 THEN 'financial_capital_intensive'
      ELSE 'services_other'
    END AS sector_proxy

  FROM with_rolling
)

-- ─────────────────────────────────────────────────────────────
-- Final output: drop intermediate columns, keep engineered features
-- ─────────────────────────────────────────────────────────────
SELECT
  * EXCEPT(row_num)
FROM with_interactions
ORDER BY firm_id, fiscal_year, fiscal_period;
