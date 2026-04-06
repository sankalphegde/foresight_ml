"""Shared utilities for the Foresight-ML Streamlit dashboard.

Provides risk classification, SHAP helpers, number formatting,
quarter utilities, and Plotly chart theming.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Risk thresholds and color palette
# ---------------------------------------------------------------------------

HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.30

COLORS = {
    "high": "#b91c1c",
    "medium": "#d97706",
    "low": "#166534",
    "high_bg": "#fee2e2",
    "medium_bg": "#fef9c3",
    "low_bg": "#dcfce7",
    "info": "#1e40af",
    "info_bg": "#dbeafe",
    "muted": "#9c9a92",
    "text": "#1a1a18",
    "border": "rgba(0,0,0,0.12)",
    "shap_positive": "#b91c1c",
    "shap_negative": "#16a34a",
}


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------


def risk_level(score: float) -> str:
    """Classify a distress probability into High / Medium / Low."""
    if score >= HIGH_RISK_THRESHOLD:
        return "High"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def risk_emoji(score: float) -> str:
    """Return colored circle emoji for a risk score."""
    if score >= HIGH_RISK_THRESHOLD:
        return "🔴"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "🟡"
    return "🟢"


def risk_color(score: float) -> str:
    """Return hex color for a risk score."""
    if score >= HIGH_RISK_THRESHOLD:
        return COLORS["high"]
    if score >= MEDIUM_RISK_THRESHOLD:
        return COLORS["medium"]
    return COLORS["low"]


def risk_badge_html(score: float) -> str:
    """Return an HTML badge for inline risk display."""
    level = risk_level(score)
    bg = COLORS[f"{level.lower()}_bg"]
    color = COLORS[level.lower()]
    return (
        f'<span style="background:{bg};color:{color};padding:3px 10px;'
        f'border-radius:20px;font-size:12px;font-weight:500">'
        f"{risk_emoji(score)} {level} risk — {score:.2f}</span>"
    )


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------


def parse_top_features_json(json_str: str) -> list[dict]:
    """Parse top_features_json into a list of dicts. Empty list on failure."""
    if pd.isna(json_str) or not json_str:
        return []
    try:
        return json.loads(json_str)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, TypeError):
        return []


def shap_color(shap_value: float) -> str:
    """Return color for SHAP value: red = increases risk, green = protective."""
    return COLORS["shap_positive"] if shap_value > 0 else COLORS["shap_negative"]


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def fmt_large_number(value: float) -> str:
    """Format large numbers: $24.2B, $3.1M, $500K."""
    if pd.isna(value):
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.1f}B"
    if abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.1f}M"
    if abs_val >= 1e3:
        return f"{sign}${abs_val / 1e3:.0f}K"
    return f"{sign}${abs_val:.0f}"


# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------


def quarter_label(year: int, period: str) -> str:
    """Human-readable quarter label: 'Q3 2025'."""
    return f"{period} {year}"


def quarter_sort_key(year: int, period: str) -> int:
    """Numeric sort key for year-quarter ordering."""
    q_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    return year * 10 + q_map.get(period, 0)


# ---------------------------------------------------------------------------
# Plotly chart theme
# ---------------------------------------------------------------------------


def apply_chart_theme(fig: Any) -> Any:
    """Apply consistent Foresight-ML theme to a Plotly figure."""
    fig.update_layout(
        font_family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        font_size=12,
        font_color=COLORS["text"],
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        legend={
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": COLORS["border"],
            "borderwidth": 1,
        },
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.06)", linecolor=COLORS["border"])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)", linecolor=COLORS["border"])
    return fig
