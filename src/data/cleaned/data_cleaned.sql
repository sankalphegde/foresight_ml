CREATE OR REPLACE TABLE `financial-distress-ew.cleaned_foresight.final_v2` AS

WITH sec_wide AS (
  SELECT
    cik,
    fiscal_year,
    fiscal_period,
    quarter_key,
    end_date,
    filed_date,
    
    -- 1. Core Accounting Tags (Used for math checks)
    MAX(CASE WHEN tag = 'Assets' THEN value END) AS Assets,
    MAX(CASE WHEN tag = 'Liabilities' THEN value END) AS Liabilities,
    MAX(CASE WHEN tag = 'StockholdersEquity' THEN value END) AS StockholdersEquity,
    
    -- 2. The Rest of the Top 40 Tags
    MAX(CASE WHEN tag = 'LiabilitiesAndStockholdersEquity' THEN value END) AS LiabilitiesAndStockholdersEquity,
    MAX(CASE WHEN tag = 'RetainedEarningsAccumulatedDeficit' THEN value END) AS RetainedEarningsAccumulatedDeficit,
    MAX(CASE WHEN tag = 'NetIncomeLoss' THEN value END) AS NetIncomeLoss,
    MAX(CASE WHEN tag = 'CashAndCashEquivalentsAtCarryingValue' THEN value END) AS CashAndCashEquivalentsAtCarryingValue,
    MAX(CASE WHEN tag = 'NetCashProvidedByUsedInOperatingActivities' THEN value END) AS NetCashProvidedByUsedInOperatingActivities,
    MAX(CASE WHEN tag = 'IncomeTaxExpenseBenefit' THEN value END) AS IncomeTaxExpenseBenefit,
    MAX(CASE WHEN tag = 'NetCashProvidedByUsedInFinancingActivities' THEN value END) AS NetCashProvidedByUsedInFinancingActivities,
    MAX(CASE WHEN tag = 'NetCashProvidedByUsedInInvestingActivities' THEN value END) AS NetCashProvidedByUsedInInvestingActivities,
    MAX(CASE WHEN tag = 'WeightedAverageNumberOfSharesOutstandingBasic' THEN value END) AS WeightedAverageNumberOfSharesOutstandingBasic,
    MAX(CASE WHEN tag = 'EarningsPerShareBasic' THEN value END) AS EarningsPerShareBasic,
    MAX(CASE WHEN tag = 'PropertyPlantAndEquipmentNet' THEN value END) AS PropertyPlantAndEquipmentNet,
    MAX(CASE WHEN tag = 'WeightedAverageNumberOfDilutedSharesOutstanding' THEN value END) AS WeightedAverageNumberOfDilutedSharesOutstanding,
    MAX(CASE WHEN tag = 'EarningsPerShareDiluted' THEN value END) AS EarningsPerShareDiluted,
    MAX(CASE WHEN tag = 'CommonStockValue' THEN value END) AS CommonStockValue,
    MAX(CASE WHEN tag = 'ShareBasedCompensation' THEN value END) AS ShareBasedCompensation,
    MAX(CASE WHEN tag = 'CommonStockSharesAuthorized' THEN value END) AS CommonStockSharesAuthorized,
    MAX(CASE WHEN tag = 'AccumulatedOtherComprehensiveIncomeLossNetOfTax' THEN value END) AS AccumulatedOtherComprehensiveIncomeLossNetOfTax,
    MAX(CASE WHEN tag = 'AssetsCurrent' THEN value END) AS AssetsCurrent,
    MAX(CASE WHEN tag = 'LiabilitiesCurrent' THEN value END) AS LiabilitiesCurrent,
    MAX(CASE WHEN tag = 'OperatingIncomeLoss' THEN value END) AS OperatingIncomeLoss,
    MAX(CASE WHEN tag = 'CommonStockSharesIssued' THEN value END) AS CommonStockSharesIssued,
    MAX(CASE WHEN tag = 'CommonStockParOrStatedValuePerShare' THEN value END) AS CommonStockParOrStatedValuePerShare,
    MAX(CASE WHEN tag = 'ComprehensiveIncomeNetOfTax' THEN value END) AS ComprehensiveIncomeNetOfTax,
    MAX(CASE WHEN tag = 'PaymentsToAcquirePropertyPlantAndEquipment' THEN value END) AS PaymentsToAcquirePropertyPlantAndEquipment,
    MAX(CASE WHEN tag = 'CommonStockSharesOutstanding' THEN value END) AS CommonStockSharesOutstanding,
    MAX(CASE WHEN tag = 'Goodwill' THEN value END) AS Goodwill,
    MAX(CASE WHEN tag = 'OtherAssetsNoncurrent' THEN value END) AS OtherAssetsNoncurrent,
    MAX(CASE WHEN tag = 'AccountsPayableCurrent' THEN value END) AS AccountsPayableCurrent,
    MAX(CASE WHEN tag = 'InterestExpense' THEN value END) AS InterestExpense,
    MAX(CASE WHEN tag = 'OtherLiabilitiesNoncurrent' THEN value END) AS OtherLiabilitiesNoncurrent,
    MAX(CASE WHEN tag = 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest' THEN value END) AS IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest,
    MAX(CASE WHEN tag = 'AccountsReceivableNetCurrent' THEN value END) AS AccountsReceivableNetCurrent,
    MAX(CASE WHEN tag = 'PreferredStockSharesAuthorized' THEN value END) AS PreferredStockSharesAuthorized,
    MAX(CASE WHEN tag = 'IncreaseDecreaseInAccountsReceivable' THEN value END) AS IncreaseDecreaseInAccountsReceivable,
    MAX(CASE WHEN tag = 'DepreciationDepletionAndAmortization' THEN value END) AS DepreciationDepletionAndAmortization,
    MAX(CASE WHEN tag = 'OtherNonoperatingIncomeExpense' THEN value END) AS OtherNonoperatingIncomeExpense,
    MAX(CASE WHEN tag = 'ProfitLoss' THEN value END) AS ProfitLoss
    
  FROM `financial-distress-ew.foresight_raw.sec_long`
  GROUP BY cik, fiscal_year, fiscal_period, quarter_key, end_date, filed_date
),

fred_wide AS (
  SELECT
    CONCAT(CAST(EXTRACT(YEAR FROM date) AS STRING), '_Q', CAST(EXTRACT(QUARTER FROM date) AS STRING)) AS quarter_key,
    MAX(CASE WHEN series_id = 'BAMLC0A4CBBB' THEN value END) AS BBB_spread,
    MAX(CASE WHEN series_id = 'CPIAUCSL' THEN value END) AS CPI,
    MAX(CASE WHEN series_id = 'DFF' THEN value END) AS FedFundsRate,
    MAX(CASE WHEN series_id = 'GDP' THEN value END) AS GDP,
    MAX(CASE WHEN series_id = 'UNRATE' THEN value END) AS UnemploymentRate,
    MAX(CASE WHEN series_id = 'VIXCLS' THEN value END) AS VIX
  FROM `financial-distress-ew.foresight_raw.fred_timeseries`
  GROUP BY quarter_key
),

joined_data AS (
  SELECT
    s.*,
    f.BBB_spread, f.CPI, f.FedFundsRate, f.GDP, f.UnemploymentRate, f.VIX
  FROM sec_wide s
  LEFT JOIN fred_wide f ON s.quarter_key = f.quarter_key
),

accounting_fixed AS (
  SELECT
    *,
    COALESCE(Assets, Liabilities + StockholdersEquity) AS fixed_Assets,
    COALESCE(Liabilities, Assets - StockholdersEquity) AS fixed_Liabilities,
    COALESCE(StockholdersEquity, Assets - Liabilities) AS fixed_Equity
  FROM joined_data
)

SELECT
  cik,
  fiscal_year,
  fiscal_period,
  quarter_key,
  end_date,
  filed_date,

  -- 1. Core Financial zero-imputation (Using the fixed accounting data)
  IFNULL(fixed_Assets, 0) AS Assets,
  IFNULL(fixed_Liabilities, 0) AS Liabilities,
  IFNULL(fixed_Equity, 0) AS StockholdersEquity,

  -- 2. Remaining Financial zero-imputation (The Rest of the Top 40 tags)
  IFNULL(LiabilitiesAndStockholdersEquity, 0) AS LiabilitiesAndStockholdersEquity,
  IFNULL(RetainedEarningsAccumulatedDeficit, 0) AS RetainedEarningsAccumulatedDeficit,
  IFNULL(NetIncomeLoss, 0) AS NetIncomeLoss,
  IFNULL(CashAndCashEquivalentsAtCarryingValue, 0) AS CashAndCashEquivalentsAtCarryingValue,
  IFNULL(NetCashProvidedByUsedInOperatingActivities, 0) AS NetCashProvidedByUsedInOperatingActivities,
  IFNULL(IncomeTaxExpenseBenefit, 0) AS IncomeTaxExpenseBenefit,
  IFNULL(NetCashProvidedByUsedInFinancingActivities, 0) AS NetCashProvidedByUsedInFinancingActivities,
  IFNULL(NetCashProvidedByUsedInInvestingActivities, 0) AS NetCashProvidedByUsedInInvestingActivities,
  IFNULL(WeightedAverageNumberOfSharesOutstandingBasic, 0) AS WeightedAverageNumberOfSharesOutstandingBasic,
  IFNULL(EarningsPerShareBasic, 0) AS EarningsPerShareBasic,
  IFNULL(PropertyPlantAndEquipmentNet, 0) AS PropertyPlantAndEquipmentNet,
  IFNULL(WeightedAverageNumberOfDilutedSharesOutstanding, 0) AS WeightedAverageNumberOfDilutedSharesOutstanding,
  IFNULL(EarningsPerShareDiluted, 0) AS EarningsPerShareDiluted,
  IFNULL(CommonStockValue, 0) AS CommonStockValue,
  IFNULL(ShareBasedCompensation, 0) AS ShareBasedCompensation,
  IFNULL(CommonStockSharesAuthorized, 0) AS CommonStockSharesAuthorized,
  IFNULL(AccumulatedOtherComprehensiveIncomeLossNetOfTax, 0) AS AccumulatedOtherComprehensiveIncomeLossNetOfTax,
  IFNULL(AssetsCurrent, 0) AS AssetsCurrent,
  IFNULL(LiabilitiesCurrent, 0) AS LiabilitiesCurrent,
  IFNULL(OperatingIncomeLoss, 0) AS OperatingIncomeLoss,
  IFNULL(CommonStockSharesIssued, 0) AS CommonStockSharesIssued,
  IFNULL(CommonStockParOrStatedValuePerShare, 0) AS CommonStockParOrStatedValuePerShare,
  IFNULL(ComprehensiveIncomeNetOfTax, 0) AS ComprehensiveIncomeNetOfTax,
  IFNULL(PaymentsToAcquirePropertyPlantAndEquipment, 0) AS PaymentsToAcquirePropertyPlantAndEquipment,
  IFNULL(CommonStockSharesOutstanding, 0) AS CommonStockSharesOutstanding,
  IFNULL(Goodwill, 0) AS Goodwill,
  IFNULL(OtherAssetsNoncurrent, 0) AS OtherAssetsNoncurrent,
  IFNULL(AccountsPayableCurrent, 0) AS AccountsPayableCurrent,
  IFNULL(InterestExpense, 0) AS InterestExpense,
  IFNULL(OtherLiabilitiesNoncurrent, 0) AS OtherLiabilitiesNoncurrent,
  IFNULL(IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest, 0) AS IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest,
  IFNULL(AccountsReceivableNetCurrent, 0) AS AccountsReceivableNetCurrent,
  IFNULL(PreferredStockSharesAuthorized, 0) AS PreferredStockSharesAuthorized,
  IFNULL(IncreaseDecreaseInAccountsReceivable, 0) AS IncreaseDecreaseInAccountsReceivable,
  IFNULL(DepreciationDepletionAndAmortization, 0) AS DepreciationDepletionAndAmortization,
  IFNULL(OtherNonoperatingIncomeExpense, 0) AS OtherNonoperatingIncomeExpense,
  IFNULL(ProfitLoss, 0) AS ProfitLoss,

  -- Macro Forward-Fill & Back-Fill (Partitioned chronologically per company)
  COALESCE(
    BBB_spread,
    LAST_VALUE(BBB_spread IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    FIRST_VALUE(BBB_spread IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
  ) AS BBB_spread,
  
  COALESCE(
    CPI,
    LAST_VALUE(CPI IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    FIRST_VALUE(CPI IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
  ) AS CPI,

  COALESCE(
    FedFundsRate,
    LAST_VALUE(FedFundsRate IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    FIRST_VALUE(FedFundsRate IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
  ) AS FedFundsRate,

  COALESCE(
    GDP,
    LAST_VALUE(GDP IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    FIRST_VALUE(GDP IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
  ) AS GDP,

  COALESCE(
    UnemploymentRate,
    LAST_VALUE(UnemploymentRate IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    FIRST_VALUE(UnemploymentRate IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
  ) AS UnemploymentRate,

  COALESCE(
    VIX,
    LAST_VALUE(VIX IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    FIRST_VALUE(VIX IGNORE NULLS) OVER (PARTITION BY cik ORDER BY filed_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
  ) AS VIX

FROM accounting_fixed;

-- EXPORT TO A NEW FOLDER IN CLOUD STORAGE
EXPORT DATA OPTIONS(
  uri='gs://financial-distress-data/cleaned_data/final_v2/train_*.parquet',
  format='PARQUET',
  overwrite=true,
  compression='SNAPPY'
) AS
SELECT * FROM `financial-distress-ew.cleaned_foresight.final_v2`;