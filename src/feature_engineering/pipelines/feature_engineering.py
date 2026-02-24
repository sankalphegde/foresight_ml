"""Feature Engineering Module.

===========================
Computes ~42 engineered features from cleaned financial statement data.
All functions are pure (no side effects) and individually testable.

Feature Categories:
  1. Financial Ratios     (13 features) — liquidity, leverage, profitability, efficiency
  2. Growth Rates          (8 features) — YoY % change per company
  3. Rolling Statistics   (12 features) — 4Q and 8Q rolling mean + std
  4. Z-Score & Interactions (5 features) — Altman Z approximation, cash burn, intensities
  5. Macro Interactions    (3 features) — macro × financial cross-terms
  6. Size Bucketing        (1 feature)  — asset-based quartile bucket
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Division guarded against zero denominators. Returns NaN where denom==0."""
    return np.where(denominator == 0, np.nan, numerator / denominator)


def clip_outliers(df: pd.DataFrame, columns: list, n_std: float = 5.0) -> pd.DataFrame:
    """Clip feature values to ±n_std standard deviations from the mean."""
    for col in columns:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                lower = mean - n_std * std
                upper = mean + n_std * std
                clipped = df[col].clip(lower=lower, upper=upper)
                n_clipped = (df[col] != clipped).sum()
                if n_clipped > 0:
                    logger.debug(f"Clipped {n_clipped} values in '{col}'")
                df[col] = clipped
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1. Financial Ratios (13 features)
# ═══════════════════════════════════════════════════════════════════════════

def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 13 financial ratios across liquidity, leverage, profitability, and efficiency.

    All ratios use safe_divide() to return NaN instead of inf/error
    when the denominator is zero (e.g., a company with zero liabilities
    would produce NaN for debt_to_equity rather than division error).
    """
    # ── Liquidity Ratios ──
    # Current Ratio: measures ability to pay short-term obligations
    df["current_ratio"] = safe_divide(df["AssetsCurrent"], df["LiabilitiesCurrent"])

    # Quick Ratio: liquid assets only (excludes inventory which may not sell quickly)
    df["quick_ratio"] = safe_divide(
        df["AssetsCurrent"] - df["InventoryNet"], df["LiabilitiesCurrent"]
    )

    # Cash Ratio: most conservative — only cash vs short-term debt
    df["cash_ratio"] = safe_divide(
        df["CashAndCashEquivalentsAtCarryingValue"], df["LiabilitiesCurrent"]
    )

    # ── Leverage Ratios ──
    # Debt-to-Equity: total leverage — how much debt per dollar of equity
    df["debt_to_equity"] = safe_divide(df["Liabilities"], df["StockholdersEquity"])

    # Debt-to-Assets: what fraction of assets is financed by debt
    df["debt_to_assets"] = safe_divide(df["Liabilities"], df["Assets"])

    # Interest Coverage: can the company pay its interest from operations?
    # Values < 1.5 are typically considered risky.
    df["interest_coverage"] = safe_divide(
        df["OperatingIncomeLoss"], df["InterestExpense"]
    )

    # ── Profitability Ratios ──
    # Gross Margin: revenue retention after direct costs
    df["gross_margin"] = safe_divide(df["GrossProfit"], df["Revenues"])

    # Operating Margin: revenue retention after operating expenses
    df["operating_margin"] = safe_divide(df["OperatingIncomeLoss"], df["Revenues"])

    # Net Margin: bottom-line profitability
    df["net_margin"] = safe_divide(df["NetIncomeLoss"], df["Revenues"])

    # ROA: how efficiently assets generate profit
    df["roa"] = safe_divide(df["NetIncomeLoss"], df["Assets"])

    # ROE: return generated for shareholders
    df["roe"] = safe_divide(df["NetIncomeLoss"], df["StockholdersEquity"])

    # ── Efficiency Ratios ──
    # Asset Turnover: revenue generated per dollar of assets
    df["asset_turnover"] = safe_divide(df["Revenues"], df["Assets"])

    # ── Cash Flow Ratios ──
    # Cash Flow to Debt: operating cash flow relative to total debt
    df["cash_flow_to_debt"] = safe_divide(
        df["NetCashProvidedByUsedInOperatingActivities"], df["Liabilities"]
    )

    logger.info("Computed 13 financial ratios.")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Growth Rates (8 features)
# ═══════════════════════════════════════════════════════════════════════════

GROWTH_COLUMNS = {
    "Revenues": "revenue_growth_yoy",
    "Assets": "assets_growth_yoy",
    "NetIncomeLoss": "net_income_growth_yoy",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cf_growth_yoy",
    "Liabilities": "liabilities_growth_yoy",
    "ResearchAndDevelopmentExpense": "rd_growth_yoy",
    "SellingGeneralAndAdministrativeExpense": "sga_growth_yoy",
    "GrossProfit": "gross_profit_growth_yoy",
}


def compute_growth_rates(df: pd.DataFrame, lag: int = 4) -> pd.DataFrame:
    """Compute year-over-year growth rates per company.

    Uses pct_change(lag) where lag=4 quarters for YoY comparison.
    This captures the trajectory of financial health — a deteriorating
    revenue growth trend is a strong distress signal.
    """
    # Support both old (cik) and new (firm_id) column names
    id_col = "firm_id" if "firm_id" in df.columns else "cik"
    df = df.sort_values([id_col, "fiscal_year", "fiscal_period"]).reset_index(drop=True)

    for source_col, target_col in GROWTH_COLUMNS.items():
        if source_col in df.columns:
            df[target_col] = df.groupby(id_col)[source_col].pct_change(periods=lag)

    logger.info(f"Computed {len(GROWTH_COLUMNS)} YoY growth rates (lag={lag}Q).")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. Rolling Statistics (12 features)
# ═══════════════════════════════════════════════════════════════════════════

ROLLING_FEATURES = [
    "current_ratio",
    "debt_to_equity",
    "net_margin",
    "roa",
    "cash_flow_to_debt",
    "revenue_growth_yoy",
]


def compute_rolling_stats(
    df: pd.DataFrame, windows: list[int] = None
) -> pd.DataFrame:
    """Compute rolling mean and std for key features over specified windows.

    Rolling statistics smooth out quarterly noise and capture trends:
    - Rolling MEAN shows the trend direction
    - Rolling STD shows volatility (high volatility = instability = risk)

    Windows of 4Q (1 year) and 8Q (2 years) capture short and medium-term trends.
    """
    if windows is None:
        windows = [4, 8]

    id_col = "firm_id" if "firm_id" in df.columns else "cik"
    df = df.sort_values([id_col, "fiscal_year", "fiscal_period"]).reset_index(drop=True)

    for feature in ROLLING_FEATURES:
        if feature not in df.columns:
            continue
        for w in windows:
            mean_col = f"{feature}_rolling_{w}q_mean"
            std_col = f"{feature}_rolling_{w}q_std"

            grouped = df.groupby(id_col)[feature]
            df[mean_col] = grouped.transform(
                lambda x, w=w: x.rolling(window=w, min_periods=1).mean()
            )
            df[std_col] = grouped.transform(
                lambda x, w=w: x.rolling(window=w, min_periods=2).std()
            )

    n_features = len(ROLLING_FEATURES) * len(windows) * 2
    logger.info(
        f"Computed {n_features} rolling statistics "
        f"(features={len(ROLLING_FEATURES)}, windows={windows})."
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. Z-Score & Interaction Terms (5 features)
# ═══════════════════════════════════════════════════════════════════════════

def compute_zscore_and_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite and interaction features.

    - altman_z_approx: Approximation of the classic Altman Z-score using
      available fields. The original formula uses market cap (not available),
      so we approximate with book equity. A Z < 1.8 signals distress risk.

    - cash_burn_rate: Quarter-over-quarter change in cash relative to
      operating expenses. Negative values mean the company is burning cash
      faster than spending on operations — a liquidity red flag.

    - leverage_x_margin: Interaction between debt burden and profitability.
      High leverage + low margin = extreme financial stress.

    - rd_intensity: R&D spend as % of revenue — identifies capital-intensive
      tech/pharma companies that may have different distress profiles.

    - sga_intensity: SGA spend as % of revenue — high values may indicate
      operational inefficiency.
    """
    # Altman Z-score approximation
    # Original: Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    # X1 = Working Capital / Total Assets
    # X2 = Retained Earnings / Total Assets
    # X3 = EBIT / Total Assets (we use OperatingIncomeLoss)
    # X4 = Book Equity / Total Liabilities (proxy for Market Value / Debt)
    # X5 = Sales / Total Assets
    x1 = safe_divide(df["AssetsCurrent"] - df["LiabilitiesCurrent"], df["Assets"])
    x2 = safe_divide(df["RetainedEarningsAccumulatedDeficit"], df["Assets"])
    x3 = safe_divide(df["OperatingIncomeLoss"], df["Assets"])
    x4 = safe_divide(df["StockholdersEquity"], df["Liabilities"])
    x5 = safe_divide(df["Revenues"], df["Assets"])
    df["altman_z_approx"] = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    # Cash burn rate: How fast is cash depleting relative to expenses?
    id_col = "firm_id" if "firm_id" in df.columns else "cik"
    df = df.sort_values([id_col, "fiscal_year", "fiscal_period"]).reset_index(drop=True)
    cash_change = df.groupby(id_col)["CashAndCashEquivalentsAtCarryingValue"].diff()
    operating_expenses = (
        df["CostOfGoodsAndServicesSold"]
        + df["SellingGeneralAndAdministrativeExpense"]
        + df["ResearchAndDevelopmentExpense"]
    )
    df["cash_burn_rate"] = safe_divide(cash_change, operating_expenses)

    # Leverage × Margin interaction (stress indicator)
    df["leverage_x_margin"] = df["debt_to_equity"] * df["operating_margin"]

    # R&D intensity
    df["rd_intensity"] = safe_divide(
        df["ResearchAndDevelopmentExpense"], df["Revenues"]
    )

    # SGA intensity
    df["sga_intensity"] = safe_divide(
        df["SellingGeneralAndAdministrativeExpense"], df["Revenues"]
    )

    logger.info("Computed 5 Z-score and interaction features.")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Macro Interaction Features (3 features)
# ═══════════════════════════════════════════════════════════════════════════

def compute_macro_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction terms between macroeconomic indicators and company-level ratios.

    These capture how macro conditions amplify or dampen firm-level risk:
    - fed_rate_x_leverage: Rising rates hurt highly leveraged companies more
    - unemployment_x_margin: Higher unemployment pressures margins
    - inflation_x_cash_ratio: Inflation erodes purchasing power of cash holdings
    """
    if "fed_funds" in df.columns and "debt_to_equity" in df.columns:
        df["fed_rate_x_leverage"] = df["fed_funds"] * df["debt_to_equity"]

    if "unemployment" in df.columns and "net_margin" in df.columns:
        df["unemployment_x_margin"] = df["unemployment"] * df["net_margin"]

    if "inflation" in df.columns and "cash_ratio" in df.columns:
        df["inflation_x_cash_ratio"] = df["inflation"] * df["cash_ratio"]

    logger.info("Computed 3 macro interaction features.")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 6. Size Bucketing (1 feature)
# ═══════════════════════════════════════════════════════════════════════════

def compute_size_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Assign companies to size quartiles based on total Assets.

    Labels: small (Q1), mid (Q2), large (Q3), mega (Q4).
    This enables stratified bias analysis — distress models may
    perform differently across company sizes.
    """
    df["company_size_bucket"] = pd.qcut(
        df["Assets"],
        q=4,
        labels=["small", "mid", "large", "mega"],
        duplicates="drop",
    )
    logger.info("Computed company_size_bucket (4 quartiles).")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 7. Sector Proxy (1 feature)
# ═══════════════════════════════════════════════════════════════════════════

def compute_sector_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a sector proxy from financial profile characteristics.

    Since sector/industry codes are not in the dataset, we approximate using:
    - High R&D intensity (>15% of revenue) → 'tech_pharma'
    - High inventory-to-assets ratio (>20%) → 'manufacturing_retail'
    - High interest expense ratio (>5% of revenue) → 'financial_capital_intensive'
    - Otherwise → 'services_other'

    This is an approximation, not a definitive classification.
    """
    rd_ratio = safe_divide(df["ResearchAndDevelopmentExpense"], df["Revenues"])
    inv_ratio = safe_divide(df["InventoryNet"], df["Assets"])
    int_ratio = safe_divide(df["InterestExpense"], df["Revenues"])

    conditions = [
        pd.Series(rd_ratio) > 0.15,
        pd.Series(inv_ratio) > 0.20,
        pd.Series(int_ratio) > 0.05,
    ]
    choices = ["tech_pharma", "manufacturing_retail", "financial_capital_intensive"]
    df["sector_proxy"] = np.select(conditions, choices, default="services_other")

    logger.info(
        f"Computed sector_proxy. Distribution: "
        f"{df['sector_proxy'].value_counts().to_dict()}"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

# All engineered feature column names (for clipping and downstream reference)
ENGINEERED_FEATURES = [
    # Ratios
    "current_ratio", "quick_ratio", "cash_ratio",
    "debt_to_equity", "debt_to_assets", "interest_coverage",
    "gross_margin", "operating_margin", "net_margin", "roa", "roe",
    "asset_turnover", "cash_flow_to_debt",
    # Growth
    "revenue_growth_yoy", "assets_growth_yoy", "net_income_growth_yoy",
    "operating_cf_growth_yoy", "liabilities_growth_yoy",
    "rd_growth_yoy", "sga_growth_yoy", "gross_profit_growth_yoy",
    # Z-score & Interactions
    "altman_z_approx", "cash_burn_rate", "leverage_x_margin",
    "rd_intensity", "sga_intensity",
    # Macro interactions
    "fed_rate_x_leverage", "unemployment_x_margin", "inflation_x_cash_ratio",
]


def engineer_features(
    df: pd.DataFrame,
    rolling_windows: list[int] = None,
    growth_lag: int = 4,
    clip_std: float = 5.0,
) -> pd.DataFrame:
    """Main entry point: run all feature engineering steps in order.

    Pipeline:
      1. Financial ratios (13 features)
      2. Growth rates (8 features)
      3. Rolling statistics (12 features)
      4. Z-score & interaction terms (5 features)
      5. Macro interactions (3 features)
      6. Size bucketing (1 feature)
      7. Sector proxy (1 feature)
      8. Outlier clipping

    Parameters:
      df: Cleaned DataFrame (output of data_cleaning.clean_data)
      rolling_windows: Window sizes for rolling stats (default: [4, 8])
      growth_lag: Quarters for YoY growth (default: 4)
      clip_std: Clip features beyond ±N std (default: 5.0)

    Returns:
      DataFrame with all original + engineered columns.
    """
    logger.info(f"Starting feature engineering. Input shape: {df.shape}")

    df = compute_financial_ratios(df)
    df = compute_growth_rates(df, lag=growth_lag)
    df = compute_rolling_stats(df, windows=rolling_windows)
    df = compute_zscore_and_interactions(df)
    df = compute_macro_interactions(df)
    df = compute_size_bucket(df)
    df = compute_sector_proxy(df)

    # Clip outliers on all numeric engineered features
    all_engineered = [c for c in ENGINEERED_FEATURES if c in df.columns]
    # Also include rolling stat columns
    rolling_cols = [c for c in df.columns if "_rolling_" in c]
    all_engineered.extend(rolling_cols)
    df = clip_outliers(df, all_engineered, n_std=clip_std)

    # Replace any remaining inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Feature engineering complete. Output shape: {df.shape}")
    return df
