"""Visualizations Module
======================
Generates all charts for feature analysis and bias analysis.
Outputs 15 PNG files to the specified output directory.

Charts 1-7:  Feature Distribution & Correlation
Charts 8-15: Bias Analysis
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/pipeline use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Style Configuration ──────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("viridis", 8)
FIG_DPI = 150
TITLE_SIZE = 14
LABEL_SIZE = 11


def _save_fig(fig: plt.Figure, output_dir: str, filename: str) -> str:
    """Save figure and close it. Returns the saved path."""
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1: Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list[str], output_dir: str) -> str:
    """Full correlation matrix of engineered features, clustered."""
    cols = [c for c in feature_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=TITLE_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)
    return _save_fig(fig, output_dir, "01_correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2: Top-20 Correlated Feature Pairs
# ═══════════════════════════════════════════════════════════════════════════


def plot_top_correlations(df: pd.DataFrame, feature_cols: list[str], output_dir: str) -> str:
    """Bar chart of the 20 most correlated feature pairs."""
    cols = [c for c in feature_cols if c in df.columns]
    corr = df[cols].corr()

    # Get upper triangle pairs
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append(
                {
                    "pair": f"{corr.columns[i]}\n× {corr.columns[j]}",
                    "correlation": corr.iloc[i, j],
                    "abs_corr": abs(corr.iloc[i, j]),
                }
            )

    pairs_df = pd.DataFrame(pairs).nlargest(20, "abs_corr")

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in pairs_df["correlation"]]
    ax.barh(range(len(pairs_df)), pairs_df["correlation"], color=colors)
    ax.set_yticks(range(len(pairs_df)))
    ax.set_yticklabels(pairs_df["pair"], fontsize=8)
    ax.set_xlabel("Correlation Coefficient", fontsize=LABEL_SIZE)
    ax.set_title(
        "Top 20 Most Correlated Feature Pairs",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    return _save_fig(fig, output_dir, "02_top_correlations.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3: Feature Distribution Grid
# ═══════════════════════════════════════════════════════════════════════════


def plot_feature_distributions(df: pd.DataFrame, feature_cols: list[str], output_dir: str) -> str:
    """Histogram + KDE for every engineered feature."""
    cols = [c for c in feature_cols if c in df.columns]
    n_cols = 4
    n_rows = (len(cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        if len(data) > 0:
            sns.histplot(data, kde=True, ax=ax, color=PALETTE[i % len(PALETTE)], bins=30)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)

    # Hide empty subplots
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature Distributions (Histogram + KDE)",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return _save_fig(fig, output_dir, "03_feature_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 4: Box Plot Grid by Category
# ═══════════════════════════════════════════════════════════════════════════


def plot_ratio_boxplots(df: pd.DataFrame, output_dir: str) -> str:
    """Box plots for financial ratios grouped by category."""
    categories = {
        "Liquidity": ["current_ratio", "quick_ratio", "cash_ratio"],
        "Leverage": ["debt_to_equity", "debt_to_assets", "interest_coverage"],
        "Profitability": ["gross_margin", "operating_margin", "net_margin", "roa", "roe"],
        "Efficiency & CF": ["asset_turnover", "cash_flow_to_debt"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (cat_name, cat_cols) in enumerate(categories.items()):
        ax = axes[idx]
        available = [c for c in cat_cols if c in df.columns]
        if available:
            plot_data = df[available].melt(var_name="Ratio", value_name="Value")
            sns.boxplot(
                data=plot_data,
                x="Ratio",
                y="Value",
                hue="Ratio",
                ax=ax,
                palette="Set2",
                showfliers=False,
                legend=False,
            )
            ax.set_title(cat_name, fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=30, labelsize=9)
            ax.set_xlabel("")

    fig.suptitle(
        "Financial Ratios by Category",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save_fig(fig, output_dir, "04_ratio_boxplots.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 5: Pairplot (Key Features)
# ═══════════════════════════════════════════════════════════════════════════


def plot_pairplot(df: pd.DataFrame, feature_cols: list[str], output_dir: str) -> str:
    """Scatter matrix for top 6 features by variance."""
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) == 0:
        return ""

    # Pick top 6 by variance
    variances = df[cols].var().dropna().nlargest(6)
    top_cols = variances.index.tolist()

    sample = df[top_cols].dropna().sample(n=min(500, len(df)), random_state=42)

    g = sns.pairplot(
        sample,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 15},
        diag_kws={"fill": True},
    )
    g.figure.suptitle(
        "Pairplot — Top 6 Features by Variance",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )
    path = os.path.join(output_dir, "05_pairplot.png")
    g.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(g.figure)
    logger.info(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Chart 6: Missing Data Heatmap (Pre-Imputation)
# ═══════════════════════════════════════════════════════════════════════════


def plot_missing_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Heatmap showing NaN pattern across rows × columns (pre-imputation)."""
    null_matrix = df.isnull().astype(int)

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(
        null_matrix.T,
        cbar=False,
        cmap=["#2ecc71", "#e74c3c"],
        ax=ax,
        yticklabels=True,
    )
    ax.set_title(
        "Missing Data Pattern (Red = Missing)",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.set_xlabel("Row Index", fontsize=LABEL_SIZE)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    return _save_fig(fig, output_dir, "06_missing_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 7: Missing Data Bar Chart
# ═══════════════════════════════════════════════════════════════════════════


def plot_missing_bar(df: pd.DataFrame, output_dir: str) -> str:
    """Per-column null percentage bar chart."""
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]

    if len(null_pct) == 0:
        # Create a chart showing no missing data
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(
            0.5,
            0.5,
            "No missing values detected",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title(
            "Missing Data by Column (Post-Imputation)",
            fontsize=TITLE_SIZE,
            fontweight="bold",
        )
        return _save_fig(fig, output_dir, "07_missing_bar.png")

    fig, ax = plt.subplots(figsize=(12, max(4, len(null_pct) * 0.4)))
    colors = ["#e74c3c" if v > 50 else "#f39c12" if v > 10 else "#3498db" for v in null_pct]
    ax.barh(range(len(null_pct)), null_pct.values, color=colors)
    ax.set_yticks(range(len(null_pct)))
    ax.set_yticklabels(null_pct.index, fontsize=9)
    ax.set_xlabel("Missing %", fontsize=LABEL_SIZE)
    ax.set_title(
        "Missing Data by Column",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.invert_yaxis()

    # Add percentage labels
    for i, v in enumerate(null_pct.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)

    return _save_fig(fig, output_dir, "07_missing_bar.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 8: Slice Sample Counts
# ═══════════════════════════════════════════════════════════════════════════


def plot_slice_sample_counts(bias_report: pd.DataFrame, output_dir: str) -> str:
    """Bar chart of sample sizes per slice across all dimensions."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    dimensions = bias_report["dimension"].unique()

    for idx, dim in enumerate(dimensions[:4]):
        ax = axes[idx] if idx < len(axes) else axes[-1]
        dim_data = bias_report[bias_report["dimension"] == dim]
        ax.bar(
            dim_data["slice"],
            dim_data["sample_count"],
            color=sns.color_palette("Set2", len(dim_data)),
        )
        ax.set_title(dim.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.set_ylabel("Sample Count")

        # Add count labels on bars
        for bar_idx, count in enumerate(dim_data["sample_count"]):
            ax.text(
                bar_idx,
                count + count * 0.02,
                str(int(count)),
                ha="center",
                fontsize=9,
            )

    fig.suptitle(
        "Sample Distribution Across Slices",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save_fig(fig, output_dir, "08_slice_sample_counts.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 9: Feature Distribution by Size Bucket (KDE)
# ═══════════════════════════════════════════════════════════════════════════


def plot_features_by_size(df: pd.DataFrame, key_features: list[str], output_dir: str) -> str:
    """Overlaid KDE plots for key ratios split by company_size_bucket."""
    if "company_size_bucket" not in df.columns:
        return ""

    features = [f for f in key_features[:6] if f in df.columns]
    n_rows = (len(features) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        for bucket in df["company_size_bucket"].unique():
            data = df[df["company_size_bucket"] == bucket][feat].dropna()
            if len(data) > 5:
                sns.kdeplot(data, ax=ax, label=bucket, fill=True, alpha=0.3, warn_singular=False)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature Distributions by Company Size",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save_fig(fig, output_dir, "09_features_by_size.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 10: Feature Distribution by Time Period (Violin)
# ═══════════════════════════════════════════════════════════════════════════


def plot_features_by_time(
    df: pd.DataFrame,
    key_features: list[str],
    time_split_year: int,
    output_dir: str,
) -> str:
    """Pre vs Post time period comparison using violin plots."""
    df = df.copy()
    df["time_period"] = np.where(
        df["fiscal_year"] < time_split_year,
        f"Pre-{time_split_year}",
        f"Post-{time_split_year}",
    )

    features = [f for f in key_features[:6] if f in df.columns]
    n_rows = (len(features) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        plot_data = df[[feat, "time_period"]].dropna()
        if len(plot_data) > 0:
            sns.violinplot(
                data=plot_data,
                x="time_period",
                y=feat,
                ax=ax,
                hue="time_period",
                palette=["#3498db", "#e74c3c"],
                inner="box",
                cut=0,
                legend=False,
            )
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("")

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Feature Distributions: Pre vs Post {time_split_year}",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save_fig(fig, output_dir, "10_features_by_time.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 11: Feature Distribution by Sector Proxy (Box)
# ═══════════════════════════════════════════════════════════════════════════


def plot_features_by_sector(df: pd.DataFrame, key_features: list[str], output_dir: str) -> str:
    """Box plots per sector proxy for key financial ratios."""
    if "sector_proxy" not in df.columns:
        return ""

    features = [f for f in key_features[:6] if f in df.columns]
    n_rows = (len(features) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        plot_data = df[[feat, "sector_proxy"]].dropna()
        if len(plot_data) > 0:
            sns.boxplot(
                data=plot_data,
                x="sector_proxy",
                y=feat,
                ax=ax,
                hue="sector_proxy",
                palette="Set3",
                showfliers=False,
                legend=False,
            )
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.set_xlabel("")

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature Distributions by Sector Proxy",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save_fig(fig, output_dir, "11_features_by_sector.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 12: Macro Regime Comparison
# ═══════════════════════════════════════════════════════════════════════════


def plot_macro_regime_comparison(
    df: pd.DataFrame,
    key_features: list[str],
    fed_funds_threshold: float,
    output_dir: str,
) -> str:
    """Side-by-side distributions for high vs low fed_funds regime."""
    if "fed_funds" not in df.columns:
        return ""

    df = df.copy()
    df["macro_regime"] = np.where(df["fed_funds"] <= fed_funds_threshold, "Low Rate", "High Rate")

    features = [f for f in key_features[:6] if f in df.columns]
    n_rows = (len(features) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        for regime in ["Low Rate", "High Rate"]:
            data = df[df["macro_regime"] == regime][feat].dropna()
            if len(data) > 5:
                sns.kdeplot(data, ax=ax, label=regime, fill=True, alpha=0.3)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Feature Distributions by Macro Regime (Fed Funds ≤/> {fed_funds_threshold}%)",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save_fig(fig, output_dir, "12_macro_regime_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 13: PSI Heatmap
# ═══════════════════════════════════════════════════════════════════════════


def plot_psi_heatmap(drift_matrices: dict[str, pd.DataFrame], output_dir: str) -> str:
    """PSI heatmap across all feature × slice combinations."""
    # Combine all drift matrices
    all_drift = []
    for dim, drift_df in drift_matrices.items():
        if not drift_df.empty:
            drift_df = drift_df.copy()
            drift_df["dimension"] = dim
            all_drift.append(drift_df)

    if not all_drift:
        return ""

    combined = pd.concat(all_drift, ignore_index=True)
    combined["slice_pair"] = combined["slice_a"] + " vs " + combined["slice_b"]

    # Pivot for heatmap
    pivot = combined.pivot_table(
        index="slice_pair",
        columns="feature",
        values="psi",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(max(16, len(pivot.columns) * 0.8), max(6, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "PSI"},
        vmin=0,
        vmax=0.5,
    )
    ax.set_title(
        "Population Stability Index (PSI) Across Slices",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    return _save_fig(fig, output_dir, "13_psi_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 14: Outlier Concentration by Slice
# ═══════════════════════════════════════════════════════════════════════════


def plot_outlier_concentration(
    bias_report: pd.DataFrame, key_features: list[str], output_dir: str
) -> str:
    """Stacked bar chart showing % of outliers in each slice."""
    outlier_cols = [
        f"{f}_outlier_rate" for f in key_features if f"{f}_outlier_rate" in bias_report.columns
    ]

    if not outlier_cols:
        return ""

    fig, ax = plt.subplots(figsize=(14, 7))

    labels = bias_report["dimension"] + " / " + bias_report["slice"]
    x = range(len(labels))
    bottom = np.zeros(len(labels))

    colors = sns.color_palette("tab20", len(outlier_cols))
    for i, col in enumerate(outlier_cols[:10]):  # Show max 10 features
        values = bias_report[col].fillna(0).values * 100
        feat_name = col.replace("_outlier_rate", "")
        ax.bar(x, values, bottom=bottom, label=feat_name, color=colors[i], width=0.7)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Outlier Rate (%)")
    ax.set_title(
        "Outlier Concentration by Slice",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    return _save_fig(fig, output_dir, "14_outlier_concentration.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 15: Missing Rate by Slice
# ═══════════════════════════════════════════════════════════════════════════


def plot_missing_by_slice(
    bias_report: pd.DataFrame, key_features: list[str], output_dir: str
) -> str:
    """Heatmap of missing data rates per feature per slice (pre-imputation)."""
    missing_cols = [
        f"{f}_missing_rate" for f in key_features if f"{f}_missing_rate" in bias_report.columns
    ]

    if not missing_cols:
        return ""

    labels = bias_report["dimension"] + " / " + bias_report["slice"]
    missing_data = bias_report[missing_cols].copy()
    missing_data.index = labels
    missing_data.columns = [c.replace("_missing_rate", "") for c in missing_cols]

    fig, ax = plt.subplots(
        figsize=(max(14, len(missing_data.columns) * 0.8), max(5, len(labels) * 0.5))
    )
    sns.heatmap(
        missing_data * 100,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Missing %"},
    )
    ax.set_title(
        "Missing Data Rate by Slice (%)",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    return _save_fig(fig, output_dir, "15_missing_by_slice.png")


# ═══════════════════════════════════════════════════════════════════════════
# Label Bias Visualizations
# ═══════════════════════════════════════════════════════════════════════════


def plot_label_feature_comparison(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: str,
) -> str:
    """Horizontal bar chart comparing feature means between healthy (label=0)
    and distressed (label=1) firms.
    """
    if "distress_label" not in df.columns:
        return ""

    valid_features = [f for f in feature_columns if f in df.columns][:20]
    if not valid_features:
        return ""

    df_h = df[df["distress_label"] == 0]
    df_d = df[df["distress_label"] == 1]

    means_h = [df_h[f].mean() for f in valid_features]
    means_d = [df_d[f].mean() for f in valid_features]

    fig, ax = plt.subplots(figsize=(12, max(6, len(valid_features) * 0.4)))
    y = np.arange(len(valid_features))
    bar_height = 0.35

    ax.barh(
        y - bar_height / 2,
        means_h,
        bar_height,
        label="Healthy (label=0)",
        color="#2ecc71",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.barh(
        y + bar_height / 2,
        means_d,
        bar_height,
        label="Distressed (label=1)",
        color="#e74c3c",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(valid_features, fontsize=9)
    ax.set_xlabel("Mean Value", fontsize=10)
    ax.set_title("Feature Means: Healthy vs Distressed Firms", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(output_dir, "16_label_feature_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_distress_rate_by_slice(
    df: pd.DataFrame,
    output_dir: str,
) -> str:
    """Grouped bar chart showing distress rate (%) across company size
    and sector proxy slices.
    """
    if "distress_label" not in df.columns:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Company Size
    if "company_size_bucket" in df.columns:
        rates_size = df.groupby("company_size_bucket")["distress_label"].mean() * 100
        rates_size = rates_size.sort_values(ascending=False)
        colors = ["#e74c3c" if v > 2 else "#f39c12" if v > 1 else "#2ecc71" for v in rates_size]
        bars = axes[0].bar(
            rates_size.index, rates_size.values, color=colors, edgecolor="white", linewidth=0.5
        )
        axes[0].set_title("Distress Rate by Company Size", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Distress Rate (%)", fontsize=10)
        axes[0].set_xlabel("Company Size Bucket", fontsize=10)
        for bar, val in zip(bars, rates_size.values, strict=False):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        axes[0].axhline(
            y=df["distress_label"].mean() * 100,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Overall: {df['distress_label'].mean() * 100:.2f}%",
        )
        axes[0].legend(fontsize=8)

    # Sector Proxy
    if "sector_proxy" in df.columns:
        rates_sector = df.groupby("sector_proxy")["distress_label"].mean() * 100
        rates_sector = rates_sector.sort_values(ascending=False)
        colors = ["#e74c3c" if v > 2 else "#f39c12" if v > 1 else "#2ecc71" for v in rates_sector]
        bars = axes[1].bar(
            rates_sector.index, rates_sector.values, color=colors, edgecolor="white", linewidth=0.5
        )
        axes[1].set_title("Distress Rate by Sector", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Distress Rate (%)", fontsize=10)
        axes[1].tick_params(axis="x", rotation=30, labelsize=8)
        for bar, val in zip(bars, rates_sector.values, strict=False):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        axes[1].axhline(
            y=df["distress_label"].mean() * 100, color="gray", linestyle="--", linewidth=1
        )

    plt.suptitle("Label Distribution Across Slices", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "17_distress_rate_by_slice.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_disparate_impact(
    df: pd.DataFrame,
    output_dir: str,
) -> str:
    """Horizontal bar chart showing Disparate Impact Ratio (DIR) for each
    slice, with the 0.8 and 1.25 fairness thresholds marked.
    """
    if "distress_label" not in df.columns:
        return ""

    dim_configs = []
    if "company_size_bucket" in df.columns:
        rates = df.groupby("company_size_bucket")["distress_label"].mean()
        ref_rate = rates.max()
        for group, rate in rates.items():
            dim_configs.append(("Company Size", group, rate / ref_rate if ref_rate > 0 else 0))

    if "sector_proxy" in df.columns:
        rates = df.groupby("sector_proxy")["distress_label"].mean()
        ref_rate = rates.max()
        for group, rate in rates.items():
            dim_configs.append(("Sector", group, rate / ref_rate if ref_rate > 0 else 0))

    if not dim_configs:
        return ""

    labels = [f"{dim}: {group}" for dim, group, _ in dim_configs]
    dir_values = [d for _, _, d in dim_configs]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    y = np.arange(len(labels))
    colors = ["#e74c3c" if d < 0.8 or d > 1.25 else "#2ecc71" for d in dir_values]

    ax.barh(y, dir_values, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

    # Fairness threshold lines
    ax.axvline(x=0.8, color="#e67e22", linewidth=2, linestyle="--", label="80% Rule Lower Bound")
    ax.axvline(x=1.0, color="gray", linewidth=1, linestyle="-", alpha=0.5)
    ax.axvline(x=1.25, color="#e67e22", linewidth=2, linestyle="--", label="125% Rule Upper Bound")

    # Fair zone shading
    ax.axvspan(0.8, 1.25, alpha=0.08, color="#2ecc71", label="Fair Zone")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Disparate Impact Ratio", fontsize=10)
    ax.set_title("Disparate Impact Analysis (80% Rule)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.invert_yaxis()

    # Value annotations
    for i, v in enumerate(dir_values):
        ax.text(
            v + 0.02, i, f"{v:.3f}", va="center", fontsize=9, color="#e74c3c" if v < 0.8 else "#333"
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "18_disparate_impact.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard HTML Generator
# ═══════════════════════════════════════════════════════════════════════════


def generate_dashboard_html(
    output_dir: str,
    row_count: int = 0,
    col_count: int = 0,
    feature_count: int = 0,
    alert_count: int = 0,
) -> str:
    """Generate a self-contained HTML dashboard that displays all 15 plots.
    Images are referenced via relative paths so the HTML works from the
    plots directory without a server.

    Called automatically at the end of generate_all_visualizations().
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Chart metadata: (filename, title, badge, description, section, layout)
    charts = [
        (
            "01_correlation_heatmap.png",
            "Correlation Heatmap",
            "Chart 1",
            "Full correlation matrix of all engineered features. Blue = negative, Red = positive. Use to identify multicollinearity and feature reduction candidates.",
            "correlation",
            "full",
        ),
        (
            "02_top_correlations.png",
            "Top 20 Most Correlated Feature Pairs",
            "Chart 2",
            "The 20 feature pairs with strongest absolute correlation. Green = positive, Red = negative. High |ρ| > 0.9 pairs are candidates for removal.",
            "correlation",
            "full",
        ),
        (
            "03_feature_distributions.png",
            "Feature Distribution Grid (Histogram + KDE)",
            "Chart 3",
            "Histogram with kernel density estimate for every engineered feature. Look for skewness, bimodality, and heavy tails that may require transformation.",
            "distribution",
            "full",
        ),
        (
            "04_ratio_boxplots.png",
            "Financial Ratios by Category",
            "Chart 4",
            "Box plots grouped by Liquidity, Leverage, Profitability, and Efficiency. Outlier whiskers excluded for clarity.",
            "distribution",
            "half",
        ),
        (
            "05_pairplot.png",
            "Pairplot — Top 6 Features by Variance",
            "Chart 5",
            "Scatter matrix of the 6 highest-variance features. Diagonal shows KDE, off-diagonal shows pairwise scatter.",
            "distribution",
            "half",
        ),
        (
            "06_missing_heatmap.png",
            "Missing Data Pattern (Pre-Imputation)",
            "Chart 6",
            "Red = missing, Green = present. Reveals structural missingness patterns — e.g., macro columns sharing the same null rows.",
            "missing",
            "half",
        ),
        (
            "07_missing_bar.png",
            "Missing Data by Column",
            "Chart 7",
            "Per-column null percentage. Red = >50%, Orange = 10-50%, Blue = <10%. The 65.8% null in macro columns is handled via forward-fill.",
            "missing",
            "half",
        ),
        (
            "08_slice_sample_counts.png",
            "Sample Distribution Across Slices",
            "Chart 8",
            "Sample sizes per slice across all 4 bias dimensions. Imbalanced slices can cause biased model performance.",
            "bias",
            "full",
        ),
        (
            "09_features_by_size.png",
            "Features by Company Size (KDE)",
            "Chart 9",
            "Overlaid density plots split by company size bucket. Significant separation indicates size-dependent feature behavior.",
            "bias",
            "half",
        ),
        (
            "10_features_by_time.png",
            "Features by Time Period (Violin)",
            "Chart 10",
            "Pre-2016 vs Post-2016 comparison. Temporal stability means the model is less likely to suffer from concept drift.",
            "bias",
            "half",
        ),
        (
            "11_features_by_sector.png",
            "Features by Sector Proxy (Box)",
            "Chart 11",
            "Box plots per inferred sector. Different sectors have fundamentally different financial profiles.",
            "bias",
            "half",
        ),
        (
            "12_macro_regime_comparison.png",
            "Macro Regime Comparison (KDE)",
            "Chart 12",
            "High vs Low federal funds rate regime. Overlapping distributions indicate feature robustness to rate environments.",
            "bias",
            "half",
        ),
        (
            "13_psi_heatmap.png",
            "Population Stability Index (PSI) Heatmap",
            "Chart 13",
            "PSI measures distributional shift. PSI < 0.10 = stable, 0.10–0.25 = moderate, > 0.25 = significant shift.",
            "bias",
            "full",
        ),
        (
            "14_outlier_concentration.png",
            "Outlier Concentration by Slice",
            "Chart 14",
            "Fraction of each slice's data that are outliers (beyond ±3σ). Concentrated outliers in one slice can bias training.",
            "bias",
            "half",
        ),
        (
            "15_missing_by_slice.png",
            "Missing Rate by Slice",
            "Chart 15",
            "Per-feature missing data rates across bias slices. Non-uniform missingness could introduce systematic bias.",
            "bias",
            "half",
        ),
        (
            "16_label_feature_comparison.png",
            "Feature Means: Healthy vs Distressed",
            "Chart 16",
            "Compares average feature values between non-distressed (green) and distressed (red) firms. Large differences indicate strong predictive signals or potential leakage.",
            "bias",
            "full",
        ),
        (
            "17_distress_rate_by_slice.png",
            "Distress Rate by Slice",
            "Chart 17",
            "Bar charts showing distress label rate (%) across company size buckets and sector proxies. Color-coded: red >2%, amber >1%, green <1%. Dashed line = overall rate.",
            "bias",
            "full",
        ),
        (
            "18_disparate_impact.png",
            "Disparate Impact Analysis (80% Rule)",
            "Chart 18",
            "Disparate Impact Ratio (DIR) per slice. Green bars fall within the fair zone (0.8–1.25). Red bars violate the 80% rule, indicating potential unfair bias against that group.",
            "bias",
            "full",
        ),
    ]

    # Build chart cards HTML
    sections: dict[str, list[tuple[str, str]]] = {
        "correlation": [],
        "distribution": [],
        "missing": [],
        "bias": [],
    }
    for fname, title, badge, desc, section, layout in charts:
        card_html = f'''<div class="card {"full-width" if layout == "full" else ""}">
          <div class="card-header"><h3>{title}</h3><span class="card-badge">{badge}</span></div>
          <div class="card-body"><img src="{fname}" alt="{title}" onclick="openLightbox(this)"></div>
          <div class="card-description">{desc}</div>
        </div>'''
        sections[section].append((card_html, layout))

    def _render_section(items: list[tuple[str, str]]) -> str:
        """Group half-width cards into grid-2 divs, leave full-width cards standalone."""
        html_parts: list[str] = []
        half_buffer: list[str] = []
        for card_html, layout in items:
            if layout == "full":
                if half_buffer:
                    html_parts.append('<div class="grid-2">' + "\n".join(half_buffer) + "</div>")
                    half_buffer = []
                html_parts.append(card_html)
            else:
                half_buffer.append(card_html)
        if half_buffer:
            html_parts.append('<div class="grid-2">' + "\n".join(half_buffer) + "</div>")
        return "\n".join(html_parts)

    section_meta = {
        "correlation": (
            "📊",
            "corr",
            "Feature Correlations",
            "Identify redundant and highly correlated feature pairs",
        ),
        "distribution": (
            "📈",
            "dist",
            "Feature Distributions",
            "Shape, spread, and outliers of each engineered feature",
        ),
        "missing": (
            "⚠️",
            "miss",
            "Missing Data Analysis",
            "Pre-imputation null patterns and column-level missingness rates",
        ),
        "bias": (
            "🔍",
            "bias",
            "Bias Analysis",
            "Feature distributions across company size, sector, time period, and macro regime",
        ),
    }

    sections_html = ""
    for sec_key in ["correlation", "distribution", "missing", "bias"]:
        icon, css_class, title, subtitle = section_meta[sec_key]
        cards_html = _render_section(sections[sec_key])
        sections_html += f'''<div class="section" data-section="{sec_key}">
      <div class="section-header">
        <div class="section-icon {css_class}">{icon}</div>
        <div><div class="section-title">{title}</div><div class="section-subtitle">{subtitle}</div></div>
      </div>
      {cards_html}
    </div>\n'''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Financial Distress Pipeline — Visualization Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{--bg-primary:#0f1117;--bg-card:#1a1d27;--border:#2a2e3d;--accent:#6c63ff;--accent-glow:rgba(108,99,255,0.25);--text-primary:#e8eaf0;--text-secondary:#9ca3b4;--text-muted:#6b7280;--radius:16px}}
    body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg-primary);color:var(--text-primary);line-height:1.6;min-height:100vh}}
    .header{{background:linear-gradient(180deg,rgba(108,99,255,0.08) 0%,transparent 100%);border-bottom:1px solid var(--border);padding:2.5rem 2rem 2rem;text-align:center;position:sticky;top:0;z-index:100;backdrop-filter:blur(20px)}}
    .header h1{{font-size:1.75rem;font-weight:700;background:linear-gradient(135deg,#6c63ff,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.02em}}
    .header p{{color:var(--text-secondary);font-size:0.9rem;margin-top:0.4rem}}
    .stats-bar{{display:flex;justify-content:center;gap:2rem;margin-top:1.2rem;flex-wrap:wrap}}
    .stat{{display:flex;align-items:center;gap:0.5rem;font-size:0.82rem;color:var(--text-secondary)}}
    .stat-value{{font-weight:600;color:var(--text-primary);font-size:0.9rem}}
    .stat-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
    .stat-dot.green{{background:#10b981}}.stat-dot.blue{{background:#3b82f6}}.stat-dot.amber{{background:#f59e0b}}.stat-dot.purple{{background:#8b5cf6}}
    .nav{{display:flex;justify-content:center;gap:0.5rem;padding:1rem 2rem;background:rgba(15,17,23,0.9);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:99;backdrop-filter:blur(12px);flex-wrap:wrap}}
    .nav-btn{{padding:0.5rem 1rem;border-radius:8px;border:1px solid transparent;background:transparent;color:var(--text-secondary);font-family:inherit;font-size:0.82rem;font-weight:500;cursor:pointer;transition:all 0.2s ease}}
    .nav-btn:hover{{background:var(--bg-card);color:var(--text-primary);border-color:var(--border)}}
    .nav-btn.active{{background:var(--accent);color:white;border-color:var(--accent);box-shadow:0 2px 12px var(--accent-glow)}}
    .content{{max-width:1400px;margin:0 auto;padding:2rem}}
    .section{{margin-bottom:3rem}}
    .section-header{{display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding-bottom:0.75rem;border-bottom:1px solid var(--border)}}
    .section-icon{{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0}}
    .section-icon.corr{{background:linear-gradient(135deg,rgba(108,99,255,0.2),rgba(59,130,246,0.2))}}
    .section-icon.dist{{background:linear-gradient(135deg,rgba(16,185,129,0.2),rgba(6,182,212,0.2))}}
    .section-icon.miss{{background:linear-gradient(135deg,rgba(245,158,11,0.2),rgba(239,68,68,0.2))}}
    .section-icon.bias{{background:linear-gradient(135deg,rgba(139,92,246,0.2),rgba(236,72,153,0.2))}}
    .section-title{{font-size:1.15rem;font-weight:600}}
    .section-subtitle{{font-size:0.82rem;color:var(--text-muted);margin-top:0.1rem}}
    .card{{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;transition:all 0.3s ease;margin-bottom:1.5rem}}
    .card:hover{{border-color:rgba(108,99,255,0.3);box-shadow:0 8px 32px rgba(0,0,0,0.5),0 0 0 1px rgba(108,99,255,0.1);transform:translateY(-2px)}}
    .card-header{{padding:1rem 1.25rem;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}}
    .card-header h3{{font-size:0.95rem;font-weight:600}}
    .card-badge{{font-size:0.7rem;font-weight:500;padding:0.2rem 0.6rem;border-radius:6px;background:rgba(108,99,255,0.15);color:#a5a0ff;white-space:nowrap}}
    .card-body{{padding:1rem;display:flex;justify-content:center;background:rgba(0,0,0,0.15)}}
    .card-body img{{max-width:100%;height:auto;border-radius:8px;cursor:pointer;transition:transform 0.3s ease}}
    .card-body img:hover{{transform:scale(1.01)}}
    .card-description{{padding:0.75rem 1.25rem;font-size:0.82rem;color:var(--text-secondary);border-top:1px solid var(--border);line-height:1.5}}
    .grid-2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(550px,1fr));gap:1.5rem}}
    .grid-2 .card{{margin-bottom:0}}
    .full-width .card-body img{{max-height:700px}}
    .lightbox{{display:none;position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,0.92);backdrop-filter:blur(8px);justify-content:center;align-items:center;padding:2rem;cursor:zoom-out;animation:fadeIn 0.2s ease}}
    .lightbox.active{{display:flex}}
    .lightbox img{{max-width:95vw;max-height:92vh;border-radius:12px;box-shadow:0 16px 64px rgba(0,0,0,0.6)}}
    .lightbox-close{{position:absolute;top:1.5rem;right:2rem;font-size:1.5rem;color:var(--text-secondary);cursor:pointer;width:40px;height:40px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:rgba(255,255,255,0.08);border:none;transition:background 0.2s}}
    .lightbox-close:hover{{background:rgba(255,255,255,0.15);color:white}}
    .footer{{text-align:center;padding:2rem;color:var(--text-muted);font-size:0.78rem;border-top:1px solid var(--border)}}
    @keyframes fadeIn{{from{{opacity:0}}to{{opacity:1}}}}
    @media(max-width:768px){{.grid-2{{grid-template-columns:1fr}}.header h1{{font-size:1.3rem}}.content{{padding:1rem}}}}
    ::-webkit-scrollbar{{width:8px}}::-webkit-scrollbar-track{{background:var(--bg-primary)}}::-webkit-scrollbar-thumb{{background:var(--border);border-radius:4px}}
  </style>
</head>
<body>
  <div class="header">
    <h1>Financial Distress Pipeline — Visualization Dashboard</h1>
    <p>Feature Engineering &amp; Bias Analysis Results</p>
    <div class="stats-bar">
      <div class="stat"><span class="stat-dot green"></span><span class="stat-value">{row_count:,}</span> rows</div>
      <div class="stat"><span class="stat-dot blue"></span><span class="stat-value">{col_count}</span> columns</div>
      <div class="stat"><span class="stat-dot amber"></span><span class="stat-value">{feature_count}</span> engineered features</div>
      <div class="stat"><span class="stat-dot purple"></span><span class="stat-value">{alert_count}</span> drift alerts</div>
    </div>
  </div>
  <nav class="nav" id="nav">
    <button class="nav-btn active" data-target="all">All</button>
    <button class="nav-btn" data-target="correlation">Correlations</button>
    <button class="nav-btn" data-target="distribution">Distributions</button>
    <button class="nav-btn" data-target="missing">Missing Data</button>
    <button class="nav-btn" data-target="bias">Bias Analysis</button>
  </nav>
  <div class="content">{sections_html}</div>
  <div class="footer">Auto-generated by pipeline on {timestamp}</div>
  <div class="lightbox" id="lightbox" onclick="closeLightbox()">
    <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
    <img id="lightbox-img" src="" alt="Enlarged view">
  </div>
  <script>
    const navBtns=document.querySelectorAll('.nav-btn'),sections=document.querySelectorAll('.section');
    navBtns.forEach(b=>b.addEventListener('click',()=>{{navBtns.forEach(x=>x.classList.remove('active'));b.classList.add('active');const t=b.dataset.target;sections.forEach(s=>{{if(t==='all'||s.dataset.section===t){{s.style.display='';s.style.animation='fadeIn 0.3s ease'}}else{{s.style.display='none'}}}})}}));
    const lb=document.getElementById('lightbox'),lbImg=document.getElementById('lightbox-img');
    function openLightbox(i){{lbImg.src=i.src;lbImg.alt=i.alt;lb.classList.add('active');document.body.style.overflow='hidden'}}
    function closeLightbox(){{lb.classList.remove('active');document.body.style.overflow=''}}
    document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeLightbox()}});
  </script>
</body>
</html>"""

    path = os.path.join(output_dir, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Saved dashboard: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def generate_all_visualizations(
    df_raw: pd.DataFrame,
    df_engineered: pd.DataFrame,
    bias_report: pd.DataFrame,
    analysis_details: dict,
    feature_columns: list[str],
    key_features: list[str],
    output_dir: str,
    time_split_year: int = 2016,
    fed_funds_threshold: float = 2.0,
) -> list[str]:
    """Generate all 15 visualizations.

    Parameters:
      df_raw: Original DataFrame BEFORE imputation (for missing data charts)
      df_engineered: DataFrame AFTER feature engineering
      bias_report: Output of run_bias_analysis()
      analysis_details: Details dict from run_bias_analysis()
      feature_columns: All engineered feature column names
      key_features: Subset of features for detailed bias charts
      output_dir: Directory to save PNG files

    Returns:
      List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    logger.info(f"Generating visualizations to: {output_dir}")

    # Feature Distribution & Correlation (use engineered data)
    saved_paths.append(plot_correlation_heatmap(df_engineered, feature_columns, output_dir))
    saved_paths.append(plot_top_correlations(df_engineered, feature_columns, output_dir))
    saved_paths.append(plot_feature_distributions(df_engineered, feature_columns, output_dir))
    saved_paths.append(plot_ratio_boxplots(df_engineered, output_dir))
    saved_paths.append(plot_pairplot(df_engineered, feature_columns, output_dir))

    # Missing data charts (use RAW data before imputation)
    saved_paths.append(plot_missing_heatmap(df_raw, output_dir))
    saved_paths.append(plot_missing_bar(df_raw, output_dir))

    # Bias analysis charts (use engineered data + bias report)
    saved_paths.append(plot_slice_sample_counts(bias_report, output_dir))
    saved_paths.append(plot_features_by_size(df_engineered, key_features, output_dir))
    saved_paths.append(
        plot_features_by_time(df_engineered, key_features, time_split_year, output_dir)
    )
    saved_paths.append(plot_features_by_sector(df_engineered, key_features, output_dir))
    saved_paths.append(
        plot_macro_regime_comparison(df_engineered, key_features, fed_funds_threshold, output_dir)
    )
    saved_paths.append(plot_psi_heatmap(analysis_details.get("drift_matrices", {}), output_dir))
    saved_paths.append(plot_outlier_concentration(bias_report, key_features, output_dir))
    saved_paths.append(plot_missing_by_slice(bias_report, key_features, output_dir))

    # Label bias charts (use engineered data with distress_label)
    saved_paths.append(plot_label_feature_comparison(df_engineered, feature_columns, output_dir))
    saved_paths.append(plot_distress_rate_by_slice(df_engineered, output_dir))
    saved_paths.append(plot_disparate_impact(df_engineered, output_dir))

    saved_paths = [p for p in saved_paths if p]  # Remove empty strings

    # Generate HTML dashboard
    alert_count = len(analysis_details.get("alerts", []))
    dashboard_path = generate_dashboard_html(
        output_dir=output_dir,
        row_count=len(df_engineered),
        col_count=df_engineered.shape[1],
        feature_count=len(feature_columns),
        alert_count=alert_count,
    )
    saved_paths.append(dashboard_path)

    logger.info(f"Generated {len(saved_paths)} visualizations + dashboard.")
    return saved_paths
