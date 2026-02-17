CREATE OR REPLACE TABLE `financial-distress-ew.foresight_cleaned.final_training_set`
PARTITION BY filed_date
CLUSTER BY ticker
AS

WITH 

-- STEP 1: FIX RAW DATES (Hybrid Logic)

deduped_source AS (
  SELECT 
    cik, 
    fiscal_year, 
    fiscal_period, 
    -- [LOGIC]: Handles the "Huge Number" (Nanoseconds) vs "YYYYMMDD" issue
    CASE 
      WHEN SAFE_CAST(filed_date AS INT64) > 30000000 
        THEN DATE(TIMESTAMP_SECONDS(DIV(SAFE_CAST(filed_date AS INT64), 1000000000)))
      ELSE SAFE.PARSE_DATE('%Y%m%d', CAST(filed_date AS STRING))
    END AS filed_date,
    tag, 
    value
  FROM `financial-distress-ew.foresight_raw.sec_long`
  WHERE 
    unit = 'USD' 
    AND value IS NOT NULL
  QUALIFY ROW_NUMBER() OVER(
    PARTITION BY cik, fiscal_year, fiscal_period, tag 
    ORDER BY filed_date DESC
  ) = 1
),

-- STEP 2: PIVOT (The "Top 50" Financial Tags)

pivoted_data AS (
  SELECT * FROM (
    SELECT 
      cik, 
      fiscal_year, 
      fiscal_period, 
      MAX(filed_date) OVER(PARTITION BY cik, fiscal_year, fiscal_period) as filed_date,
      tag, 
      value 
    FROM deduped_source
    WHERE filed_date IS NOT NULL
  )
  PIVOT (
    ANY_VALUE(value) 
    FOR tag IN (
        -- --- BALANCE SHEET (ASSETS) ---
        'Assets', 
        'AssetsCurrent',
        'CashAndCashEquivalentsAtCarryingValue',
        'InventoryNet',
        'AccountsReceivableNetCurrent',
        'PropertyPlantAndEquipmentNet',
        'Goodwill',
        'IntangibleAssetsNetExcludingGoodwill',
        
        -- --- BALANCE SHEET (LIABILITIES) ---
        'Liabilities', 
        'LiabilitiesCurrent',
        'AccountsPayableCurrent',
        'LongTermDebt',
        'LongTermDebtCurrent',
        'CommitmentsAndContingencies',
        
        -- --- BALANCE SHEET (EQUITY) ---
        'StockholdersEquity',
        'RetainedEarningsAccumulatedDeficit',
        'AdditionalPaidInCapital',
        'CommonStockValue',
        'TreasuryStockValue',

        -- --- INCOME STATEMENT ---
        'Revenues',
        'CostOfGoodsAndServicesSold',
        'GrossProfit',
        'OperatingIncomeLoss',
        'NetIncomeLoss',
        'ResearchAndDevelopmentExpense',
        'SellingGeneralAndAdministrativeExpense',
        'InterestExpense',
        'IncomeTaxExpenseBenefit',
        'EarningsPerShareBasic',
        'EarningsPerShareDiluted',
        'WeightedAverageNumberOfSharesOutstandingBasic',
        
        -- --- CASH FLOW STATEMENT ---
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInInvestingActivities',
        'NetCashProvidedByUsedInFinancingActivities',
        'DepreciationDepletionAndAmortization',
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'ProceedsFromIssuanceOfCommonStock'
    )
  )
),


-- STEP 3: JOIN MACRO DATA

joined_data AS (
  SELECT 
    f.*, 
    p.ticker,
    p.fed_funds,    
    p.unemployment, 
    p.inflation     
  FROM pivoted_data f
  LEFT JOIN `financial-distress-ew.foresight_raw.panel_base` p
    ON f.cik = p.cik 
    AND f.filed_date = DATE(p.filing_date)
),


-- STEP 4: CLEANING & IMPUTATION (For ALL Tags)

final_clean AS (
  SELECT
    cik, ticker, fiscal_year, fiscal_period, filed_date,
    
    -- --- ASSETS ---
    IFNULL(Assets, AVG(Assets) OVER(PARTITION BY fiscal_year)) AS Assets,
    IFNULL(AssetsCurrent, AVG(AssetsCurrent) OVER(PARTITION BY fiscal_year)) AS AssetsCurrent,
    IFNULL(CashAndCashEquivalentsAtCarryingValue, AVG(CashAndCashEquivalentsAtCarryingValue) OVER(PARTITION BY fiscal_year)) AS CashAndCashEquivalentsAtCarryingValue,
    IFNULL(InventoryNet, AVG(InventoryNet) OVER(PARTITION BY fiscal_year)) AS InventoryNet,
    IFNULL(AccountsReceivableNetCurrent, AVG(AccountsReceivableNetCurrent) OVER(PARTITION BY fiscal_year)) AS AccountsReceivableNetCurrent,
    IFNULL(PropertyPlantAndEquipmentNet, AVG(PropertyPlantAndEquipmentNet) OVER(PARTITION BY fiscal_year)) AS PropertyPlantAndEquipmentNet,
    IFNULL(Goodwill, AVG(Goodwill) OVER(PARTITION BY fiscal_year)) AS Goodwill,
    IFNULL(IntangibleAssetsNetExcludingGoodwill, AVG(IntangibleAssetsNetExcludingGoodwill) OVER(PARTITION BY fiscal_year)) AS IntangibleAssetsNetExcludingGoodwill,

    -- --- LIABILITIES ---
    IFNULL(Liabilities, AVG(Liabilities) OVER(PARTITION BY fiscal_year)) AS Liabilities,
    IFNULL(LiabilitiesCurrent, AVG(LiabilitiesCurrent) OVER(PARTITION BY fiscal_year)) AS LiabilitiesCurrent,
    IFNULL(AccountsPayableCurrent, AVG(AccountsPayableCurrent) OVER(PARTITION BY fiscal_year)) AS AccountsPayableCurrent,
    IFNULL(LongTermDebt, AVG(LongTermDebt) OVER(PARTITION BY fiscal_year)) AS LongTermDebt,
    IFNULL(LongTermDebtCurrent, AVG(LongTermDebtCurrent) OVER(PARTITION BY fiscal_year)) AS LongTermDebtCurrent,

    -- --- EQUITY ---
    IFNULL(StockholdersEquity, AVG(StockholdersEquity) OVER(PARTITION BY fiscal_year)) AS StockholdersEquity,
    IFNULL(RetainedEarningsAccumulatedDeficit, AVG(RetainedEarningsAccumulatedDeficit) OVER(PARTITION BY fiscal_year)) AS RetainedEarningsAccumulatedDeficit,
    IFNULL(AdditionalPaidInCapital, AVG(AdditionalPaidInCapital) OVER(PARTITION BY fiscal_year)) AS AdditionalPaidInCapital,
    IFNULL(CommonStockValue, AVG(CommonStockValue) OVER(PARTITION BY fiscal_year)) AS CommonStockValue,

    -- --- INCOME ---
    IFNULL(Revenues, AVG(Revenues) OVER(PARTITION BY fiscal_year)) AS Revenues,
    IFNULL(CostOfGoodsAndServicesSold, AVG(CostOfGoodsAndServicesSold) OVER(PARTITION BY fiscal_year)) AS CostOfGoodsAndServicesSold,
    IFNULL(GrossProfit, AVG(GrossProfit) OVER(PARTITION BY fiscal_year)) AS GrossProfit,
    IFNULL(OperatingIncomeLoss, AVG(OperatingIncomeLoss) OVER(PARTITION BY fiscal_year)) AS OperatingIncomeLoss,
    IFNULL(NetIncomeLoss, AVG(NetIncomeLoss) OVER(PARTITION BY fiscal_year)) AS NetIncomeLoss,
    IFNULL(ResearchAndDevelopmentExpense, AVG(ResearchAndDevelopmentExpense) OVER(PARTITION BY fiscal_year)) AS ResearchAndDevelopmentExpense,
    IFNULL(SellingGeneralAndAdministrativeExpense, AVG(SellingGeneralAndAdministrativeExpense) OVER(PARTITION BY fiscal_year)) AS SellingGeneralAndAdministrativeExpense,
    IFNULL(InterestExpense, AVG(InterestExpense) OVER(PARTITION BY fiscal_year)) AS InterestExpense,
    IFNULL(IncomeTaxExpenseBenefit, AVG(IncomeTaxExpenseBenefit) OVER(PARTITION BY fiscal_year)) AS IncomeTaxExpenseBenefit,
    IFNULL(EarningsPerShareBasic, AVG(EarningsPerShareBasic) OVER(PARTITION BY fiscal_year)) AS EarningsPerShareBasic,
    IFNULL(EarningsPerShareDiluted, AVG(EarningsPerShareDiluted) OVER(PARTITION BY fiscal_year)) AS EarningsPerShareDiluted,
    
    -- --- CASH FLOW ---
    IFNULL(NetCashProvidedByUsedInOperatingActivities, AVG(NetCashProvidedByUsedInOperatingActivities) OVER(PARTITION BY fiscal_year)) AS NetCashProvidedByUsedInOperatingActivities,
    IFNULL(NetCashProvidedByUsedInInvestingActivities, AVG(NetCashProvidedByUsedInInvestingActivities) OVER(PARTITION BY fiscal_year)) AS NetCashProvidedByUsedInInvestingActivities,
    IFNULL(NetCashProvidedByUsedInFinancingActivities, AVG(NetCashProvidedByUsedInFinancingActivities) OVER(PARTITION BY fiscal_year)) AS NetCashProvidedByUsedInFinancingActivities,
    IFNULL(DepreciationDepletionAndAmortization, AVG(DepreciationDepletionAndAmortization) OVER(PARTITION BY fiscal_year)) AS DepreciationDepletionAndAmortization,
    
    -- Pass through Macros
    fed_funds,
    unemployment,
    inflation,

    -- Accounting Check
    CASE 
      WHEN Assets < 0 THEN 'Error: Negative Assets'
      ELSE 'Valid'
    END AS quality_check_flag

  FROM joined_data
)

SELECT * FROM final_clean;