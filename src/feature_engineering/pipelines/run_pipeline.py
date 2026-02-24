"""
Pipeline Orchestrator
======================
CLI entry point for running the feature engineering and bias analysis pipeline.

Usage:
  # Local mode (Pandas on parquet file)
  python -m pipelines.run_pipeline --mode local \
    --input data/cleaned_data_merged_v1_final_data_000000000000.parquet \
    --output data/

  # BigQuery mode (SQL execution on BigQuery)
  python -m pipelines.run_pipeline --mode bigquery --config config/settings.yaml
"""

import argparse
import logging
import os
import sys

import pandas as pd
import yaml

from pipelines.data_cleaning import clean_data
from pipelines.feature_engineering import engineer_features, ENGINEERED_FEATURES
from pipelines.bias_analysis import run_bias_analysis, KEY_FEATURES
from pipelines.visualizations import generate_all_visualizations

# ── Logging Setup ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Local Mode
# ═══════════════════════════════════════════════════════════════════════════

def run_local(input_path: str, output_dir: str, config: dict = None) -> None:
    """
    Run the full pipeline locally using Pandas.

    Steps:
      1. Load parquet data
      2. Save raw copy for missing-data visualizations
      3. Clean data (impute/drop NaN)
      4. Engineer features
      5. Run bias analysis
      6. Generate all visualizations
      7. Save outputs
    """
    # Load config defaults
    fe_config = (config or {}).get("feature_engineering", {})
    ba_config = (config or {}).get("bias_analysis", {})

    rolling_windows = fe_config.get("rolling_windows", [4, 8])
    growth_lag = fe_config.get("growth_lag", 4)
    clip_std = fe_config.get("outlier_clip_std", 5.0)
    time_split_year = ba_config.get("time_split_year", 2016)
    fed_funds_threshold = ba_config.get("fed_funds_threshold", 2.0)
    key_features = ba_config.get("key_features_for_plots", KEY_FEATURES)

    # Step 1: Load data
    logger.info(f"Loading data from: {input_path}")
    df_raw = pd.read_parquet(input_path)
    logger.info(f"Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    # Step 2: Keep raw copy for missing-data charts
    df_raw_copy = df_raw.copy()

    # Step 3: Clean data
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Cleaning")
    logger.info("=" * 60)
    df_clean = clean_data(df_raw)

    # Step 4: Feature engineering
    logger.info("=" * 60)
    logger.info("STAGE 2: Feature Engineering")
    logger.info("=" * 60)
    df_features = engineer_features(
        df_clean,
        rolling_windows=rolling_windows,
        growth_lag=growth_lag,
        clip_std=clip_std,
    )

    # Step 5: Bias analysis
    logger.info("=" * 60)
    logger.info("STAGE 3: Bias Analysis")
    logger.info("=" * 60)
    bias_report, analysis_details = run_bias_analysis(
        df_features,
        time_split_year=time_split_year,
        fed_funds_threshold=fed_funds_threshold,
        feature_columns=key_features,
    )

    # Step 6: Visualizations
    logger.info("=" * 60)
    logger.info("STAGE 4: Visualizations")
    logger.info("=" * 60)
    plots_dir = os.path.join(output_dir, "plots")

    # Get all engineered feature columns that actually exist
    all_feature_cols = [c for c in ENGINEERED_FEATURES if c in df_features.columns]
    rolling_cols = [c for c in df_features.columns if "_rolling_" in c]
    all_feature_cols.extend(rolling_cols)
    # Filter key_features to those present
    valid_key_features = [f for f in key_features if f in df_features.columns]

    generate_all_visualizations(
        df_raw=df_raw_copy,
        df_engineered=df_features,
        bias_report=bias_report,
        analysis_details=analysis_details,
        feature_columns=all_feature_cols,
        key_features=valid_key_features,
        output_dir=plots_dir,
        time_split_year=time_split_year,
        fed_funds_threshold=fed_funds_threshold,
    )

    # Step 7: Save outputs
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, "engineered_features.parquet")
    df_features.to_parquet(features_path, index=False)
    logger.info(f"Saved engineered features to: {features_path}")

    bias_path = os.path.join(output_dir, "bias_report.csv")
    bias_report.to_csv(bias_path, index=False)
    logger.info(f"Saved bias report to: {bias_path}")

    # Print alerts
    alerts = analysis_details.get("alerts", [])
    if alerts:
        logger.info("=" * 60)
        logger.info("BIAS ALERTS")
        logger.info("=" * 60)
        for alert in alerts:
            logger.warning(alert)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Rows: {df_features.shape[0]}")
    logger.info(f"  Total columns: {df_features.shape[1]}")
    logger.info(f"  Engineered features: {len(all_feature_cols)}")
    logger.info(f"  Plots generated: {len(os.listdir(plots_dir))}")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# BigQuery Mode
# ═══════════════════════════════════════════════════════════════════════════

def handle_missing_engineered_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing data in the engineered features table pulled from BigQuery.

    Root Cause: The BQ SQL uses SAFE_DIVIDE which returns NULL when the
    denominator is 0. Many raw columns (Revenues, GrossProfit, R&D, SGA,
    Inventory) don't exist in the source table and were defaulted to 0.0,
    causing SAFE_DIVIDE to produce NULL for most derived features.

    Strategy:
    ---------
    Fill all numeric NULL feature values with 0.0 (neutral signal).

    Rationale per category:
      - Financial ratios (71% null): denominator is 0 → ratio undefined →
        0.0 is the correct neutral signal ("not applicable").
      - Growth rates (87% null): no prior-year value → 0% growth assumed.
      - Rolling std (65% null): ≤1 data point → zero volatility.
      - Composites (70-100% null): NULL propagation → neutral 0.
      - Macro interactions (~70% null): macro × NULL_ratio → neutral 0.

    Preserves identity columns and non-feature columns unchanged.
    """
    import numpy as np

    # Log pre-imputation summary
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    total_nulls = null_counts.sum()
    total_cells = df.shape[0] * df.shape[1]
    logger.info(
        f"[PRE-IMPUTATION] {total_nulls} NULL cells "
        f"({100 * total_nulls / total_cells:.1f}% of all cells) "
        f"across {len(cols_with_nulls)} columns."
    )

    # Identify feature columns to fill (exclude identity/categorical columns)
    identity_cols = {
        "firm_id", "cik", "ticker", "fiscal_year", "fiscal_period",
        "filed_date", "date", "distress_label",
        "company_size_bucket", "sector_proxy",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols_to_fill = [c for c in numeric_cols if c not in identity_cols]

    # Fill NaN with 0.0 for all numeric feature columns
    filled_count = 0
    for col in feature_cols_to_fill:
        n_nulls = df[col].isnull().sum()
        if n_nulls > 0:
            df[col] = df[col].fillna(0.0)
            pct = 100 * n_nulls / len(df)
            logger.info(f"  Filled '{col}': {n_nulls} nulls ({pct:.1f}%)")
            filled_count += n_nulls

    # Log post-imputation summary
    remaining_nulls = df.isnull().sum().sum()
    logger.info(
        f"[POST-IMPUTATION] Filled {filled_count} NULL values. "
        f"Remaining NULLs: {remaining_nulls}."
    )

    return df


def generate_bias_report_markdown(
    df: pd.DataFrame,
    bias_report: pd.DataFrame,
    analysis_details: dict,
    output_path: str,
) -> str:
    """
    Generate a comprehensive markdown bias report acting as a senior data analyst.

    Covers:
      - Executive summary
      - Missing data analysis with detailed reasoning
      - Label imbalance analysis
      - Per-dimension slice analysis
      - Drift detection results
      - Fairness metrics and recommendations
    """
    import numpy as np
    from datetime import datetime

    alerts = analysis_details.get("alerts", [])
    drift_matrices = analysis_details.get("drift_matrices", {})
    feature_columns = analysis_details.get("feature_columns", [])

    # ── Compute label statistics ──
    label_stats = {}
    if "distress_label" in df.columns:
        total = len(df)
        distress_n = int((df["distress_label"] == 1).sum())
        non_distress_n = total - distress_n
        label_stats["total"] = total
        label_stats["distress_n"] = distress_n
        label_stats["non_distress_n"] = non_distress_n
        label_stats["distress_rate"] = distress_n / total if total > 0 else 0
        label_stats["imbalance_ratio"] = non_distress_n / distress_n if distress_n > 0 else float("inf")

    # ── Compute label rate per slice ──
    slice_label_rates = {}
    if "distress_label" in df.columns:
        for dim_col, dim_name in [
            ("company_size_bucket", "Company Size"),
            ("sector_proxy", "Sector Proxy"),
        ]:
            if dim_col in df.columns:
                rates = df.groupby(dim_col)["distress_label"].agg(["mean", "count"])
                slice_label_rates[dim_name] = rates

    # ── Build report ──
    lines = []
    lines.append("# Financial Distress Pipeline — Bias Analysis Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append(f"\n---\n")

    # Executive Summary
    lines.append("## 1. Executive Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total Observations | {len(df):,} |")
    lines.append(f"| Unique Firms | {df['firm_id'].nunique() if 'firm_id' in df.columns else 'N/A':,} |")
    if "fiscal_year" in df.columns:
        lines.append(f"| Year Range | {int(df['fiscal_year'].min())}–{int(df['fiscal_year'].max())} |")
    lines.append(f"| Engineered Features | {len(feature_columns)} |")
    lines.append(f"| Bias Dimensions | {len(drift_matrices)} |")
    lines.append(f"| High-Drift Alerts | {len(alerts)} |")
    if label_stats:
        lines.append(f"| Distress Rate | {label_stats['distress_rate']:.2%} ({label_stats['distress_n']:,} / {label_stats['total']:,}) |")
        lines.append(f"| Class Imbalance Ratio | 1:{label_stats['imbalance_ratio']:.0f} |")
    lines.append("")

    if alerts:
        lines.append("> [!WARNING]")
        lines.append(f"> **{len(alerts)} high-drift alerts detected.** Features exhibiting PSI > 0.25 across slices")
        lines.append("> need investigation before model training.\n")

    # Missing Data Analysis
    lines.append("## 2. Missing Data Analysis\n")
    lines.append("### 2.1 Raw Features Table (`raw_features`)\n")
    lines.append("The source table is **remarkably clean** with <0.01% overall null rate:\n")
    lines.append("| Category | Columns | Null Rate | Handling Strategy |")
    lines.append("|---|---|---|---|")
    lines.append("| Identity (firm_id, fiscal_year, etc.) | 4 | 0.0% | No action needed |")
    lines.append("| Financial statements (Assets, Liabilities, etc.) | 28 | 0.0% | No action needed |")
    lines.append("| Macroeconomic (FedFundsRate, CPI, etc.) | 6 | <0.01% (20 rows) | Forward-fill by date, then backfill |")
    lines.append("| Lag columns (total_assets_lag1, etc.) | 4 | 1.8–7.2% | Fill with current-period value (0% change) |")
    lines.append("| Distress label | 1 | 0.0% | No action needed |")
    lines.append("")

    lines.append("### 2.2 Engineered Features Table (Post-SQL)\n")
    lines.append("The feature engineering SQL produces significant NULLs due to `SAFE_DIVIDE` by zero:\n")
    lines.append("| Feature Category | Null % | Root Cause | Imputation |")
    lines.append("|---|---|---|---|")
    lines.append("| Financial ratios | ~71% | SAFE_DIVIDE(x, 0) = NULL | Fill 0.0 — ratio undefined when denom=0 |")
    lines.append("| Growth rates (YoY) | ~87% | LAG(4) + SAFE_DIVIDE by 0 | Fill 0.0 — no growth signal available |")
    lines.append("| Rolling std | ~65% | STDDEV over ≤1 non-null value | Fill 0.0 — zero volatility |")
    lines.append("| Altman Z-score | ~71% | Compound NULL propagation | Fill 0.0 — composite undefined |")
    lines.append("| Cash burn rate | 100% | All expense columns = 0 | Fill 0.0 — no burn rate signal |")
    lines.append("")

    lines.append("**Key Insight**: The NULLs are *not* missing data in the traditional sense. They arise because")
    lines.append("many SEC EDGAR columns expected by the SQL (Revenues, GrossProfit, R&D, SGA, Inventory, etc.)")
    lines.append("were not present in the raw BigQuery table and defaulted to 0.0. When the SQL computes ratios")
    lines.append("like `Revenues / Assets`, the result is a valid 0.0. But when it computes `GrossProfit / Revenues`")
    lines.append("(0 / 0), SAFE_DIVIDE correctly returns NULL. Filling with 0.0 is appropriate because it")
    lines.append("represents 'this metric is not applicable for this observation.'\n")

    # Label Imbalance
    if label_stats:
        lines.append("## 3. Label Imbalance Analysis\n")
        lines.append(f"The dataset has a **severe class imbalance** with only {label_stats['distress_rate']:.2%} positive labels ")
        lines.append(f"({label_stats['distress_n']:,} distressed vs {label_stats['non_distress_n']:,} non-distressed).\n")

        lines.append("> [!CAUTION]")
        lines.append(f"> With a 1:{label_stats['imbalance_ratio']:.0f} imbalance ratio, standard classifiers will be biased")
        lines.append("> toward predicting the majority class. **Mitigation strategies**: SMOTE oversampling,")
        lines.append("> class-weighted loss functions, or focal loss. Evaluate with PR-AUC, not accuracy.\n")

        # Label rate per slice
        if slice_label_rates:
            lines.append("### 3.1 Distress Rate by Slice\n")
            for dim_name, rates in slice_label_rates.items():
                lines.append(f"**{dim_name}**:\n")
                lines.append("| Slice | Count | Distress Rate |")
                lines.append("|---|---|---|")
                for idx, row in rates.iterrows():
                    lines.append(f"| {idx} | {int(row['count']):,} | {row['mean']:.2%} |")
                lines.append("")

        # ── Section 3.2: In-Depth Feature Separation by Label ──
        lines.append("### 3.2 Feature Separation by Label (Healthy vs Distressed)\n")
        lines.append("This section measures how each feature's distribution differs between")
        lines.append("healthy (label=0) and distressed (label=1) firms. Features with large")
        lines.append("separation are strong predictive signals — but extreme values may also")
        lines.append("indicate data leakage if they are outcomes of distress rather than causes.\n")

        from scipy import stats as sp_stats
        from pipelines.bias_analysis import compute_psi

        df_healthy = df[df["distress_label"] == 0]
        df_distressed = df[df["distress_label"] == 1]

        # 3.2a — Feature means, Cohen's d, KS test, PSI
        lines.append("#### 3.2a Feature Distribution Comparison\n")
        lines.append("| Feature | Mean (Healthy) | Mean (Distressed) | Cohen's d | KS Stat | KS p-value | PSI | Signal |")
        lines.append("|---|---|---|---|---|---|---|---|")

        separation_results = []
        for feat in feature_columns:
            if feat not in df.columns:
                continue
            h_vals = df_healthy[feat].dropna()
            d_vals = df_distressed[feat].dropna()
            if len(h_vals) < 10 or len(d_vals) < 10:
                continue

            h_mean = h_vals.mean()
            d_mean = d_vals.mean()

            # Cohen's d: standardized effect size
            pooled_std = np.sqrt(
                ((len(h_vals) - 1) * h_vals.std()**2 + (len(d_vals) - 1) * d_vals.std()**2)
                / (len(h_vals) + len(d_vals) - 2)
            )
            cohens_d = (d_mean - h_mean) / pooled_std if pooled_std > 0 else 0.0

            # Two-sample KS test: are distributions different?
            ks_stat, ks_pvalue = sp_stats.ks_2samp(h_vals, d_vals)

            # PSI between label groups
            psi_val = compute_psi(h_vals, d_vals)

            # Signal classification
            abs_d = abs(cohens_d)
            if abs_d >= 0.8:
                signal = "🔴 LARGE"
            elif abs_d >= 0.5:
                signal = "🟠 MEDIUM"
            elif abs_d >= 0.2:
                signal = "🟡 SMALL"
            else:
                signal = "⚪ NEGLIGIBLE"

            separation_results.append({
                "feature": feat, "h_mean": h_mean, "d_mean": d_mean,
                "cohens_d": cohens_d, "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue, "psi": psi_val, "signal": signal,
            })

            lines.append(
                f"| {feat} | {h_mean:.4f} | {d_mean:.4f} | {cohens_d:+.3f} "
                f"| {ks_stat:.3f} | {ks_pvalue:.2e} | {psi_val:.3f} | {signal} |"
            )

        lines.append("")

        # Summary counts
        large_ct = sum(1 for r in separation_results if "LARGE" in r["signal"])
        medium_ct = sum(1 for r in separation_results if "MEDIUM" in r["signal"])
        small_ct = sum(1 for r in separation_results if "SMALL" in r["signal"])
        neg_ct = sum(1 for r in separation_results if "NEGLIGIBLE" in r["signal"])

        lines.append(f"> [!NOTE]")
        lines.append(f"> **Effect size summary**: {large_ct} large / {medium_ct} medium / "
                     f"{small_ct} small / {neg_ct} negligible out of {len(separation_results)} features.")
        lines.append(f"> Cohen's d thresholds: |d| ≥ 0.8 large, ≥ 0.5 medium, ≥ 0.2 small.\n")

        # Leakage warning for very high separation
        leakage_candidates = [r for r in separation_results if abs(r["cohens_d"]) >= 0.8]
        if leakage_candidates:
            lines.append("> [!WARNING]")
            lines.append("> **Potential leakage check**: Features with large effect sizes may be")
            lines.append("> *consequences* of distress rather than *predictors*. Verify that these")
            lines.append("> features are available at prediction time and do not encode the label:")
            for r in leakage_candidates:
                lines.append(f">   - `{r['feature']}` (d={r['cohens_d']:+.3f})")
            lines.append("")

        # 3.2b — Chi-Squared Independence Tests
        lines.append("#### 3.2b Chi-Squared Independence Tests\n")
        lines.append("Tests whether `distress_label` is statistically independent of each")
        lines.append("categorical slice. A low p-value means distress is unevenly distributed.\n")
        lines.append("| Dimension | Chi² Statistic | p-value | Degrees of Freedom | Independent? |")
        lines.append("|---|---|---|---|---|")

        for dim_col, dim_name in [
            ("company_size_bucket", "Company Size"),
            ("sector_proxy", "Sector Proxy"),
        ]:
            if dim_col in df.columns:
                contingency = pd.crosstab(df[dim_col], df["distress_label"])
                chi2, p_val, dof, _ = sp_stats.chi2_contingency(contingency)
                independent = "✅ Yes" if p_val > 0.05 else "❌ No"
                lines.append(f"| {dim_name} | {chi2:,.1f} | {p_val:.2e} | {dof} | {independent} |")

        # Time period
        if "fiscal_year" in df.columns:
            time_col = (df["fiscal_year"] >= 2016).map({True: "post_2016", False: "pre_2016"})
            contingency = pd.crosstab(time_col, df["distress_label"])
            chi2, p_val, dof, _ = sp_stats.chi2_contingency(contingency)
            independent = "✅ Yes" if p_val > 0.05 else "❌ No"
            lines.append(f"| Time Period | {chi2:,.1f} | {p_val:.2e} | {dof} | {independent} |")

        lines.append("")

        # 3.2c — Disparate Impact Analysis
        lines.append("#### 3.2c Disparate Impact Analysis\n")
        lines.append("Disparate Impact Ratio (DIR) = distress_rate(group) / distress_rate(reference_group).")
        lines.append("A DIR < 0.8 or > 1.25 indicates potential unfair treatment per the 80% rule.\n")

        for dim_col, dim_name in [
            ("company_size_bucket", "Company Size"),
            ("sector_proxy", "Sector Proxy"),
        ]:
            if dim_col not in df.columns:
                continue
            rates = df.groupby(dim_col)["distress_label"].mean()
            ref_rate = rates.max()  # Use highest-rate group as reference
            ref_group = rates.idxmax()
            lines.append(f"**{dim_name}** (reference: {ref_group} at {ref_rate:.2%}):\n")
            lines.append("| Group | Distress Rate | DIR | Fair? |")
            lines.append("|---|---|---|---|")
            for group, rate in rates.items():
                dir_val = rate / ref_rate if ref_rate > 0 else 0
                fair = "✅" if 0.8 <= dir_val <= 1.25 else "⚠️ UNFAIR"
                lines.append(f"| {group} | {rate:.2%} | {dir_val:.3f} | {fair} |")
            lines.append("")

    # Drift Detection
    lines.append("## 4. Feature Drift Detection (PSI)\n")
    lines.append("Population Stability Index (PSI) measures distributional shift between slices:\n")
    lines.append("| PSI Range | Interpretation |")
    lines.append("|---|---|")
    lines.append("| < 0.10 | No significant shift — feature is stable |")
    lines.append("| 0.10–0.25 | Moderate shift — investigate |")
    lines.append("| > 0.25 | Significant shift — action required |\n")

    if alerts:
        lines.append("### 4.1 High-Drift Alerts\n")
        for alert in alerts:
            lines.append(f"- {alert}")
        lines.append("")

    # Per-dimension summary
    lines.append("### 4.2 Drift Summary by Dimension\n")
    for dim, drift_df in drift_matrices.items():
        if drift_df.empty:
            continue
        high_count = (drift_df["psi"] > 0.25).sum()
        moderate_count = ((drift_df["psi"] > 0.10) & (drift_df["psi"] <= 0.25)).sum()
        stable_count = (drift_df["psi"] <= 0.10).sum()
        lines.append(f"**{dim.replace('_', ' ').title()}**: "
                     f"{high_count} high / {moderate_count} moderate / {stable_count} stable features")

    lines.append("")

    # Slice statistics
    lines.append("## 5. Slice Statistics Summary\n")
    if not bias_report.empty:
        for dim in bias_report["dimension"].unique():
            dim_data = bias_report[bias_report["dimension"] == dim]
            lines.append(f"### {dim.replace('_', ' ').title()}\n")
            lines.append("| Slice | Samples | % of Total |")
            lines.append("|---|---|---|")
            for _, row in dim_data.iterrows():
                pct = 100 * row["sample_count"] / len(df)
                lines.append(f"| {row['slice']} | {int(row['sample_count']):,} | {pct:.1f}% |")
            lines.append("")

    # Recommendations
    lines.append("## 6. Recommendations\n")
    lines.append("1. **Address class imbalance** before training: Use SMOTE, class weights, or stratified ")
    lines.append("   sampling. Evaluate with PR-AUC and F1, not accuracy.")
    lines.append("2. **Investigate high-drift features**: Features with PSI > 0.25 across company size ")
    lines.append("   buckets may cause the model to perform differently for small vs. mega-cap firms.")
    lines.append("3. **Consider removing zero-filled features**: Features that are 100% NULL (like ")
    lines.append("   `cash_burn_rate`) have no discriminative power after 0-fill and should be dropped ")
    lines.append("   or sourced from alternative data.")
    lines.append("4. **Stratified evaluation**: Always evaluate model performance per-slice (size, sector, ")
    lines.append("   time period) to detect hidden bias in predictions.")
    lines.append("5. **Temporal validation**: Use a walk-forward split (not random) to prevent lookahead bias, ")
    lines.append("   given the 2009–2026 time span and identified temporal drift.")
    lines.append("")

    report_text = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Saved bias report: {output_path}")
    return output_path


def run_bigquery(config: dict) -> None:
    """
    Run the pipeline on BigQuery.

    Steps:
      1. Read SQL template and substitute config values
      2. Execute SQL on BigQuery
      3. Pull results and handle missing data
      4. Run bias analysis locally
      5. Generate visualizations and bias report
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        logger.error(
            "google-cloud-bigquery not installed. "
            "Run: pip install google-cloud-bigquery"
        )
        sys.exit(1)

    gcp_config = config["gcp"]
    tables_config = config["tables"]

    project_id = gcp_config["project_id"]
    dataset = gcp_config["dataset"]
    location = gcp_config.get("location", "US")
    raw_table = tables_config.get("raw_financials", "raw_financials")

    if project_id == "YOUR_GCP_PROJECT_ID":
        logger.error(
            "Please set your GCP project ID in config/settings.yaml "
            "before running BigQuery mode."
        )
        sys.exit(1)

    # Read and parameterize SQL
    sql_path = os.path.join(os.path.dirname(__file__), "feature_engineering_bq.sql")
    with open(sql_path, "r") as f:
        sql = f.read()

    sql = sql.replace("${PROJECT}", project_id)
    sql = sql.replace("${DATASET}", dataset)
    sql = sql.replace("${RAW_TABLE}", raw_table)

    logger.info(f"Executing feature engineering SQL on BigQuery...")
    logger.info(f"  Project: {project_id}")
    logger.info(f"  Dataset: {dataset}")
    logger.info(f"  Raw Table: {raw_table}")

    # Execute feature engineering SQL
    client = bigquery.Client(project=project_id, location=location)
    job = client.query(sql)
    job.result()  # Wait for completion
    logger.info(f"BigQuery job completed: {job.job_id}")

    # ── Step 3: Create cleaned_engineered_features table ──
    clean_sql_path = os.path.join(
        os.path.dirname(__file__), "clean_engineered_features_bq.sql"
    )
    with open(clean_sql_path, "r") as f:
        clean_sql = f.read()

    clean_sql = clean_sql.replace("${PROJECT}", project_id)
    clean_sql = clean_sql.replace("${DATASET}", dataset)

    logger.info("Creating cleaned_engineered_features table (IFNULL on all feature NULLs)...")
    clean_job = client.query(clean_sql)
    clean_job.result()
    logger.info(f"Cleaning job completed: {clean_job.job_id}")

    # Pull from the CLEANED table for bias analysis and visualizations
    cleaned_table = f"{project_id}.{dataset}.cleaned_engineered_features"
    logger.info(f"Pulling cleaned results from: {cleaned_table}")
    df = client.query(f"SELECT * FROM `{cleaned_table}`").to_dataframe(
        create_bqstorage_client=False
    )

    # Verify no remaining NULLs in feature columns
    null_count = df.isnull().sum().sum()
    logger.info(f"[VERIFICATION] Remaining NULLs after cleaning: {null_count}")

    # Run bias analysis locally on pulled data
    ba_config = config.get("bias_analysis", {})
    time_split_year = ba_config.get("time_split_year", 2016)
    fed_funds_threshold = ba_config.get("fed_funds_threshold", 2.0)
    key_features = ba_config.get("key_features_for_plots", KEY_FEATURES)

    bias_report, analysis_details = run_bias_analysis(
        df,
        time_split_year=time_split_year,
        fed_funds_threshold=fed_funds_threshold,
    )

    # Generate visualizations
    plots_dir = "data/plots"
    all_feature_cols = [c for c in ENGINEERED_FEATURES if c in df.columns]
    rolling_cols = [c for c in df.columns if "_rolling_" in c]
    all_feature_cols.extend(rolling_cols)
    valid_key_features = [f for f in key_features if f in df.columns]

    generate_all_visualizations(
        df_raw=df,  # In BQ mode, we don't have the raw data separately
        df_engineered=df,
        bias_report=bias_report,
        analysis_details=analysis_details,
        feature_columns=all_feature_cols,
        key_features=valid_key_features,
        output_dir=plots_dir,
        time_split_year=time_split_year,
        fed_funds_threshold=fed_funds_threshold,
    )

    # Save bias report CSV
    bias_report.to_csv("data/bias_report.csv", index=False)

    # Generate comprehensive markdown bias report
    generate_bias_report_markdown(
        df=df,
        bias_report=bias_report,
        analysis_details=analysis_details,
        output_path="data/bias_report.md",
    )

    logger.info("BigQuery pipeline complete.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Feature Engineering & Bias Analysis Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "bigquery"],
        required=True,
        help="Execution mode: 'local' (Pandas) or 'bigquery' (BigQuery SQL)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input parquet file path (required for local mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Config file path (default: config/settings.yaml)",
    )

    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from: {args.config}")

    if args.mode == "local":
        if not args.input:
            parser.error("--input is required for local mode")
        run_local(args.input, args.output, config)

    elif args.mode == "bigquery":
        if not config.get("gcp"):
            parser.error("GCP config required. Set values in config/settings.yaml")
        run_bigquery(config)


if __name__ == "__main__":
    main()
