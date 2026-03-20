"""Bias Analysis Module
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

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Population Stability Index (PSI)
# ═══════════════════════════════════════════════════════════════════════════


def compute_psi(reference: pd.Series, comparison: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    PSI measures how much a distribution has shifted:
      PSI < 0.10 → No significant shift
      PSI 0.10–0.25 → Moderate shift, investigate
      PSI > 0.25 → Significant shift, action required

    Uses equal-width bins from the reference distribution.
    """
    ref_clean = reference.dropna()
    comp_clean = comparison.dropna()

    if len(ref_clean) < 10 or len(comp_clean) < 10:
        return float("nan")

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


def compute_js_divergence(reference: pd.Series, comparison: pd.Series, bins: int = 10) -> float:
    """Compute Jensen-Shannon divergence between two distributions.
    Returns a value in [0, 1] where 0 = identical distributions.
    """
    ref_clean = reference.dropna()
    comp_clean = comparison.dropna()

    if len(ref_clean) < 10 or len(comp_clean) < 10:
        return float("nan")

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
    """Create analysis slices from the dataset.

    Returns a nested dict: {dimension_name: {slice_name: DataFrame}}.
    """
    slices = {}

    # 1. Company Size
    if "company_size_bucket" in df.columns:
        company_size_slices = {}
        for name, group in df.groupby("company_size_bucket", observed=True):
            company_size_slices[name] = group
        slices["company_size"] = company_size_slices

    # 2. Sector Proxy
    if "sector_proxy" in df.columns:
        sector_slices = {}
        for name, group in df.groupby("sector_proxy", observed=True):
            sector_slices[name] = group
        slices["sector_proxy"] = sector_slices

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
    stats: dict[str, int | float] = {
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
    """Compute PSI between all pairs of slices for each feature.
    Returns a DataFrame with (slice_pair, feature) as index.
    """
    slice_names = list(slices_dict.keys())
    results = []

    for i, name_a in enumerate(slice_names):
        for name_b in slice_names[i + 1 :]:
            for col in feature_columns:
                if col not in slices_dict[name_a].columns:
                    continue
                psi = compute_psi(slices_dict[name_a][col], slices_dict[name_b][col])
                js = compute_js_divergence(slices_dict[name_a][col], slices_dict[name_b][col])
                results.append(
                    {
                        "slice_a": name_a,
                        "slice_b": name_b,
                        "feature": col,
                        "psi": psi,
                        "js_divergence": js,
                    }
                )

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

KEY_FEATURES = [
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "debt_to_equity",
    "debt_to_assets",
    "interest_coverage",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roa",
    "roe",
    "asset_turnover",
    "cash_flow_to_debt",
    "revenue_growth_yoy",
    "net_income_growth_yoy",
    "altman_z_approx",
    "cash_burn_rate",
    "rd_intensity",
    "sga_intensity",
    "leverage_x_margin",
]


def run_bias_analysis(
    df: pd.DataFrame,
    time_split_year: int = 2016,
    fed_funds_threshold: float = 2.0,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run full bias analysis across all slicing dimensions.

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

    logger.info(f"Running bias analysis on {len(feature_columns)} features across {len(df)} rows.")

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


# ═══════════════════════════════════════════════════════════════════════════
# Model-Level Fairness Analysis  (Person 4 extension)
# ═══════════════════════════════════════════════════════════════════════════

BIAS_ALERT_THRESHOLD = 0.10  # 10 percentage-point drop triggers alert


def compute_model_fairness(
    slice_performance: pd.DataFrame,
    metrics_to_check: list[str] | None = None,
    alert_threshold: float = BIAS_ALERT_THRESHOLD,
) -> pd.DataFrame:
    """Compare per-slice model metrics against the overall metric.

    For each slice, flag if ROC-AUC or Recall@K drops more than
    ``alert_threshold`` (default 10 pp) below the overall value.

    Args:
        slice_performance: DataFrame from evaluate.py's
            ``_build_slice_performance_table`` with columns including
            ``dimension``, ``slice``, ``roc_auc``, ``recall_at_5pct``.
        metrics_to_check: Metric columns to evaluate. Defaults to
            ``["roc_auc", "recall_at_5pct"]``.
        alert_threshold: Fractional drop (0.10 = 10 pp) that triggers a
            bias alert.

    Returns:
        DataFrame with one row per (dimension, slice, metric) combination,
        including the slice value, overall value, gap, and a boolean
        ``bias_alert`` flag.
    """
    if metrics_to_check is None:
        metrics_to_check = ["roc_auc", "recall_at_5pct"]

    rows: list[dict] = []

    for metric in metrics_to_check:
        if metric not in slice_performance.columns:
            logger.warning("Metric '%s' not found in slice table; skipping.", metric)
            continue

        # Overall = weighted average across all slices (weighted by sample_count)
        valid = slice_performance.dropna(subset=[metric])
        if valid.empty or valid["sample_count"].sum() == 0:
            continue

        overall_value = float(np.average(valid[metric], weights=valid["sample_count"]))

        for _, row in valid.iterrows():
            slice_value = float(row[metric])
            gap = overall_value - slice_value
            is_alert = gap > alert_threshold

            rows.append(
                {
                    "dimension": row["dimension"],
                    "slice": row["slice"],
                    "metric": metric,
                    "slice_value": round(slice_value, 4),
                    "overall_value": round(overall_value, 4),
                    "gap": round(gap, 4),
                    "bias_alert": is_alert,
                    "sample_count": int(row["sample_count"]),
                }
            )

            if is_alert:
                logger.warning(
                    "BIAS ALERT: %s for slice %s/%s = %.4f "
                    "(overall %.4f, gap %.4f > threshold %.2f)",
                    metric,
                    row["dimension"],
                    row["slice"],
                    slice_value,
                    overall_value,
                    gap,
                    alert_threshold,
                )

    return pd.DataFrame(rows)


def suggest_threshold_adjustments(
    fairness_df: pd.DataFrame,
    base_threshold: float = 0.5,
    adjustment_step: float = 0.05,
) -> pd.DataFrame:
    """Suggest per-slice threshold adjustments for slices with bias alerts.

    For slices where recall is too low relative to overall, a lower
    decision threshold is suggested (increasing sensitivity). This is a
    simple heuristic — the exact threshold should be validated on
    held-out data.

    Args:
        fairness_df: Output of ``compute_model_fairness``.
        base_threshold: Default decision threshold.
        adjustment_step: How much to lower the threshold per alert.

    Returns:
        DataFrame with suggested thresholds for alerted slices only.
    """
    alerts = fairness_df[
        (fairness_df["bias_alert"]) & (fairness_df["metric"] == "recall_at_5pct")
    ].copy()

    if alerts.empty:
        return pd.DataFrame(
            columns=["dimension", "slice", "current_threshold", "suggested_threshold", "reason"]
        )

    suggestions = []
    for _, row in alerts.iterrows():
        # Scale adjustment by the severity of the gap
        n_steps = max(1, int(row["gap"] / 0.05))
        suggested = round(base_threshold - (adjustment_step * n_steps), 3)
        suggested = max(0.1, suggested)  # floor at 0.1

        suggestions.append(
            {
                "dimension": row["dimension"],
                "slice": row["slice"],
                "current_threshold": base_threshold,
                "suggested_threshold": suggested,
                "reason": (
                    f"recall_at_5pct gap of {row['gap']:.4f} "
                    f"exceeds {BIAS_ALERT_THRESHOLD:.0%} threshold"
                ),
            }
        )

    return pd.DataFrame(suggestions)


def generate_bias_report_md(
    feature_bias_report: pd.DataFrame,
    feature_analysis_details: dict,
    model_fairness_df: pd.DataFrame,
    threshold_suggestions: pd.DataFrame,
    output_path: str | None = None,
) -> str:
    """Generate a combined bias report in Markdown.

    Merges the existing feature-level PSI/drift analysis with the new
    model-level fairness metrics into a single ``bias_report.md``.

    Args:
        feature_bias_report: Output of ``run_bias_analysis`` (feature-level).
        feature_analysis_details: Details dict from ``run_bias_analysis``.
        model_fairness_df: Output of ``compute_model_fairness``.
        threshold_suggestions: Output of ``suggest_threshold_adjustments``.
        output_path: File path to write the report. If None, only returns
            the string.

    Returns:
        The full Markdown report as a string.
    """
    lines: list[str] = []

    lines.append("# Bias Report — Foresight-ML")
    lines.append("")
    lines.append("## 1. Feature-Level Drift Analysis")
    lines.append("")

    # Drift alerts
    alerts = feature_analysis_details.get("alerts", [])
    if alerts:
        lines.append(f"**{len(alerts)} high-drift alert(s) detected:**")
        lines.append("")
        for alert in alerts:
            lines.append(f"- {alert}")
        lines.append("")
    else:
        lines.append("No high-drift alerts detected (all PSI < 0.25).")
        lines.append("")

    # Slice sample counts
    lines.append("### Slice Sample Counts")
    lines.append("")
    if not feature_bias_report.empty:
        summary = feature_bias_report[["dimension", "slice", "sample_count"]].to_markdown(
            index=False
        )
        lines.append(summary)
    else:
        lines.append("_No feature-level slice data available._")
    lines.append("")

    # ── Model-level fairness ────────────────────────────────────────────
    lines.append("## 2. Model-Level Fairness Analysis")
    lines.append("")

    if model_fairness_df.empty:
        lines.append("_No model-level fairness data available._")
    else:
        alert_count = int(model_fairness_df["bias_alert"].sum())
        lines.append(
            f"**{alert_count} bias alert(s)** detected across "
            f"{len(model_fairness_df)} (dimension, slice, metric) checks."
        )
        lines.append("")

        lines.append("### Per-Slice Performance vs Overall")
        lines.append("")
        display_cols = [
            "dimension",
            "slice",
            "metric",
            "slice_value",
            "overall_value",
            "gap",
            "bias_alert",
        ]
        available_cols = [c for c in display_cols if c in model_fairness_df.columns]
        lines.append(model_fairness_df[available_cols].to_markdown(index=False))
        lines.append("")

        # Highlight alerts
        alerted = model_fairness_df[model_fairness_df["bias_alert"]]
        if not alerted.empty:
            lines.append("### Bias Alerts (gap > 10 pp)")
            lines.append("")
            for _, row in alerted.iterrows():
                lines.append(
                    f"- **{row['dimension']}/{row['slice']}**: "
                    f"{row['metric']} = {row['slice_value']:.4f} "
                    f"(overall {row['overall_value']:.4f}, "
                    f"gap {row['gap']:.4f})"
                )
            lines.append("")

    # ── Threshold adjustments ───────────────────────────────────────────
    lines.append("## 3. Mitigation — Threshold Adjustments")
    lines.append("")

    if threshold_suggestions.empty:
        lines.append("No threshold adjustments needed — all slices within tolerance.")
    else:
        lines.append(
            "The following per-slice threshold adjustments are suggested "
            "to improve recall for underperforming slices:"
        )
        lines.append("")
        lines.append(threshold_suggestions.to_markdown(index=False))
    lines.append("")

    lines.append("---")
    lines.append("_Report generated by Foresight-ML bias analysis pipeline._")
    lines.append("")

    report_text = "\n".join(lines)

    if output_path:
        out = pd.io.common.stringify_path(output_path)
        from pathlib import Path as _Path

        _Path(out).parent.mkdir(parents=True, exist_ok=True)
        _Path(out).write_text(report_text, encoding="utf-8")
        logger.info("Saved bias report: %s", out)

    return report_text
