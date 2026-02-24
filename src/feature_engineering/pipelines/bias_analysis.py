"""
Bias Analysis Module
=====================
Analyzes feature distributions and fairness across population slices.
Detects whether the dataset or engineered features exhibit systematic
differences that could lead to biased model predictions.

Slicing Dimensions:
  1. Company Size   — small / mid / large / mega (from asset quartiles)
  2. Sector Proxy   — tech_pharma / manufacturing_retail / financial / services
  3. Time Period    — pre-2016 vs post-2016 (regulatory + economic regime)
  4. Macro Regime   — high vs low federal funds rate
"""

import logging
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Population Stability Index (PSI)
# ═══════════════════════════════════════════════════════════════════════════

def compute_psi(reference: pd.Series, comparison: pd.Series, bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI measures how much a distribution has shifted:
      PSI < 0.10 → No significant shift
      PSI 0.10–0.25 → Moderate shift, investigate
      PSI > 0.25 → Significant shift, action required

    Uses equal-width bins from the reference distribution.
    """
    ref_clean = reference.dropna()
    comp_clean = comparison.dropna()

    if len(ref_clean) < 10 or len(comp_clean) < 10:
        return np.nan

    # Create bins from the combined range
    min_val = min(ref_clean.min(), comp_clean.min())
    max_val = max(ref_clean.max(), comp_clean.max())
    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    comp_counts, _ = np.histogram(comp_clean, bins=bin_edges)

    # Convert to proportions with smoothing to avoid log(0)
    ref_pct = (ref_counts + 1e-6) / (ref_counts.sum() + bins * 1e-6)
    comp_pct = (comp_counts + 1e-6) / (comp_counts.sum() + bins * 1e-6)

    psi = np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct))
    return float(psi)


def compute_js_divergence(
    reference: pd.Series, comparison: pd.Series, bins: int = 10
) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    Returns a value in [0, 1] where 0 = identical distributions.
    """
    ref_clean = reference.dropna()
    comp_clean = comparison.dropna()

    if len(ref_clean) < 10 or len(comp_clean) < 10:
        return np.nan

    min_val = min(ref_clean.min(), comp_clean.min())
    max_val = max(ref_clean.max(), comp_clean.max())
    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)

    ref_hist, _ = np.histogram(ref_clean, bins=bin_edges, density=True)
    comp_hist, _ = np.histogram(comp_clean, bins=bin_edges, density=True)

    # Normalize to probability distributions
    ref_prob = ref_hist / (ref_hist.sum() + 1e-10)
    comp_prob = comp_hist / (comp_hist.sum() + 1e-10)

    return float(jensenshannon(ref_prob, comp_prob))


# ═══════════════════════════════════════════════════════════════════════════
# Slice Definitions
# ═══════════════════════════════════════════════════════════════════════════

def create_slices(
    df: pd.DataFrame,
    time_split_year: int = 2016,
    fed_funds_threshold: float = 2.0,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Create analysis slices from the dataset.

    Returns a nested dict: {dimension_name: {slice_name: DataFrame}}.
    """
    slices = {}

    # 1. Company Size
    if "company_size_bucket" in df.columns:
        slices["company_size"] = {
            name: group
            for name, group in df.groupby("company_size_bucket", observed=True)
        }

    # 2. Sector Proxy
    if "sector_proxy" in df.columns:
        slices["sector_proxy"] = {
            name: group
            for name, group in df.groupby("sector_proxy", observed=True)
        }

    # 3. Time Period
    if "fiscal_year" in df.columns:
        slices["time_period"] = {
            f"pre_{time_split_year}": df[df["fiscal_year"] < time_split_year],
            f"post_{time_split_year}": df[df["fiscal_year"] >= time_split_year],
        }

    # 4. Macro Regime
    if "fed_funds" in df.columns:
        slices["macro_regime"] = {
            "low_rate": df[df["fed_funds"] <= fed_funds_threshold],
            "high_rate": df[df["fed_funds"] > fed_funds_threshold],
        }

    # 5. Distress Label
    if "distress_label" in df.columns:
        slices["distress_label"] = {
            "healthy (label=0)": df[df["distress_label"] == 0],
            "distressed (label=1)": df[df["distress_label"] == 1],
        }

    return slices


# ═══════════════════════════════════════════════════════════════════════════
# Per-Slice Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_slice_statistics(
    df_slice: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Compute summary statistics for a single slice."""
    stats = {
        "sample_count": len(df_slice),
    }

    for col in feature_columns:
        if col not in df_slice.columns:
            continue
        series = df_slice[col]
        stats[f"{col}_mean"] = series.mean()
        stats[f"{col}_std"] = series.std()
        stats[f"{col}_median"] = series.median()
        stats[f"{col}_q25"] = series.quantile(0.25)
        stats[f"{col}_q75"] = series.quantile(0.75)
        stats[f"{col}_missing_rate"] = series.isnull().mean()

        # Outlier rate (beyond ±3 std from overall mean)
        if series.std() > 0:
            mean, std = series.mean(), series.std()
            outlier_mask = (series < mean - 3 * std) | (series > mean + 3 * std)
            stats[f"{col}_outlier_rate"] = outlier_mask.mean()
        else:
            stats[f"{col}_outlier_rate"] = 0.0

    return stats


def compute_drift_matrix(
    slices_dict: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Compute PSI between all pairs of slices for each feature.
    Returns a DataFrame with (slice_pair, feature) as index.
    """
    slice_names = list(slices_dict.keys())
    results = []

    for i, name_a in enumerate(slice_names):
        for name_b in slice_names[i + 1:]:
            for col in feature_columns:
                if col not in slices_dict[name_a].columns:
                    continue
                psi = compute_psi(
                    slices_dict[name_a][col], slices_dict[name_b][col]
                )
                js = compute_js_divergence(
                    slices_dict[name_a][col], slices_dict[name_b][col]
                )
                results.append({
                    "slice_a": name_a,
                    "slice_b": name_b,
                    "feature": col,
                    "psi": psi,
                    "js_divergence": js,
                })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

KEY_FEATURES = [
    "current_ratio", "quick_ratio", "cash_ratio",
    "debt_to_equity", "debt_to_assets", "interest_coverage",
    "gross_margin", "operating_margin", "net_margin", "roa", "roe",
    "asset_turnover", "cash_flow_to_debt",
    "revenue_growth_yoy", "net_income_growth_yoy",
    "altman_z_approx", "cash_burn_rate",
    "rd_intensity", "sga_intensity",
    "leverage_x_margin",
]


def run_bias_analysis(
    df: pd.DataFrame,
    time_split_year: int = 2016,
    fed_funds_threshold: float = 2.0,
    feature_columns: list[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run full bias analysis across all slicing dimensions.

    Parameters:
      df: DataFrame with engineered features
      time_split_year: Year to split pre/post time periods
      fed_funds_threshold: Threshold for high/low macro regime
      feature_columns: Features to analyze (default: KEY_FEATURES)

    Returns:
      (bias_report_df, analysis_details)
      - bias_report_df: Summary DataFrame with per-slice statistics
      - analysis_details: Dict with slices, drift matrices, alerts
    """
    if feature_columns is None:
        feature_columns = [f for f in KEY_FEATURES if f in df.columns]

    logger.info(
        f"Running bias analysis on {len(feature_columns)} features "
        f"across {len(df)} rows."
    )

    # Create slices
    all_slices = create_slices(df, time_split_year, fed_funds_threshold)

    # Compute per-slice statistics
    report_rows = []
    for dimension, slice_dict in all_slices.items():
        for slice_name, slice_df in slice_dict.items():
            stats = analyze_slice_statistics(slice_df, feature_columns)
            stats["dimension"] = dimension
            stats["slice"] = slice_name
            report_rows.append(stats)

    bias_report = pd.DataFrame(report_rows)

    # Reorder columns
    lead_cols = ["dimension", "slice", "sample_count"]
    other_cols = [c for c in bias_report.columns if c not in lead_cols]
    bias_report = bias_report[lead_cols + other_cols]

    # Compute drift matrices per dimension
    drift_matrices = {}
    alerts = []
    for dimension, slice_dict in all_slices.items():
        drift = compute_drift_matrix(slice_dict, feature_columns)
        drift_matrices[dimension] = drift

        # Flag high-drift features
        if not drift.empty:
            high_psi = drift[drift["psi"] > 0.25]
            for _, row in high_psi.iterrows():
                alert_msg = (
                    f"⚠ HIGH DRIFT: {row['feature']} between "
                    f"{row['slice_a']} and {row['slice_b']} "
                    f"(PSI={row['psi']:.3f})"
                )
                alerts.append(alert_msg)
                logger.warning(alert_msg)

    # Log summary
    logger.info(f"Bias analysis complete. {len(report_rows)} slice-stats computed.")
    if alerts:
        logger.warning(f"{len(alerts)} high-drift alerts detected.")
    else:
        logger.info("No high-drift alerts detected.")

    analysis_details = {
        "slices": all_slices,
        "drift_matrices": drift_matrices,
        "alerts": alerts,
        "feature_columns": feature_columns,
    }

    return bias_report, analysis_details
