"""Page 1 — Company Risk Explorer.

Search by firm_id, view distress probability trend, SHAP risk drivers,
and financial snapshot for any company in the dataset.
"""

from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data.gcs_loader import (
    get_company_history,
    get_shap_for_company,
    load_labeled_panel,
    load_shap_values,
)
from src.dashboard.utils import (
    apply_chart_theme,
    fmt_large_number,
    fmt_ratio,
    quarter_label,
    quarter_sort_key,
    risk_badge_html,
    risk_color,
    risk_emoji,
    risk_level,
    shap_color,
    shap_direction_label,
    parse_top_features_json,
    COLORS,
)


def render() -> None:
    """Render the Company Risk Explorer page."""
    st.header("🔍 Company Risk Explorer")

    # ── Load data ────────────────────────────────────────────────────
    panel = load_labeled_panel()
    shap_df = load_shap_values()

    if panel.empty:
        st.error("No company data available. Check GCS connection.")
        return

    # ── Company search ───────────────────────────────────────────────
    firm_ids = sorted(panel["firm_id"].unique())

    col_search, col_sector, col_size = st.columns([3, 1, 1])

    with col_search:
        selected_firm = st.selectbox(
            "Search company (firm_id)",
            options=firm_ids,
            index=0,
            help="Select a company by its CIK / firm_id",
        )

    # Optional filters for context
    with col_sector:
        if "sector_proxy" in panel.columns:
            sectors = ["All sectors"] + sorted(panel["sector_proxy"].dropna().unique().tolist())
            st.selectbox("Sector", options=sectors, disabled=True)
        else:
            st.empty()

    with col_size:
        if "company_size_bucket" in panel.columns:
            sizes = ["All sizes"] + sorted(
                panel["company_size_bucket"].dropna().unique().tolist()
            )
            st.selectbox("Size", options=sizes, disabled=True)
        else:
            st.empty()

    if not selected_firm:
        st.info("Select a company to view risk analysis.")
        return

    # ── Company data ─────────────────────────────────────────────────
    history = get_company_history(panel, selected_firm)

    if history.empty:
        st.warning(f"No data found for firm_id: {selected_firm}")
        return

    latest = history.iloc[-1]
    latest_year = int(latest["fiscal_year"])
    latest_period = str(latest.get("fiscal_period", "Q4"))

    # Distress label as proxy risk score (0 or 1) until scores parquet exists
    if "distress_label" in latest.index:
        latest_score = float(latest["distress_label"])
    else:
        latest_score = 0.0

    # ── Company header with risk badge ───────────────────────────────
    st.markdown("---")
    hdr_left, hdr_right = st.columns([3, 1])

    with hdr_left:
        st.markdown(f"### {selected_firm}")
        meta_parts = [f"CIK {selected_firm}"]
        if "sector_proxy" in latest.index and latest.get("sector_proxy"):
            meta_parts.append(str(latest["sector_proxy"]))
        if "company_size_bucket" in latest.index and latest.get("company_size_bucket"):
            meta_parts.append(str(latest["company_size_bucket"]))
        meta_parts.append(quarter_label(latest_year, latest_period))
        st.caption(" · ".join(meta_parts))

    with hdr_right:
        st.markdown(
            risk_badge_html(latest_score),
            unsafe_allow_html=True,
        )

    # ── Distress trend chart ─────────────────────────────────────────
    st.markdown("#### Distress status — last 8 quarters")

    if "distress_label" in history.columns:
        # Take last 8 quarters
        recent = history.tail(8).copy()
        recent["quarter"] = recent.apply(
            lambda r: quarter_label(int(r["fiscal_year"]), str(r.get("fiscal_period", ""))),
            axis=1,
        )
        recent["sort_key"] = recent.apply(
            lambda r: quarter_sort_key(int(r["fiscal_year"]), str(r.get("fiscal_period", ""))),
            axis=1,
        )
        recent = recent.sort_values("sort_key")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recent["quarter"],
                y=recent["distress_label"],
                mode="lines+markers",
                line=dict(color=risk_color(latest_score), width=2),
                marker=dict(size=7),
                name="Distress label",
                hovertemplate="Quarter: %{x}<br>Distress: %{y}<extra></extra>",
            )
        )
        fig.update_yaxes(
            title_text="Distress label",
            range=[-0.1, 1.1],
            tickvals=[0, 1],
            ticktext=["Healthy", "Distressed"],
        )
        fig.update_xaxes(title_text="")
        fig.update_layout(height=250, showlegend=False)
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Distress label not available for trend chart.")

    # ── Two columns: SHAP drivers + financial snapshot ───────────────
    col_shap, col_snapshot = st.columns(2)

    # ── SHAP risk drivers ────────────────────────────────────────────
    with col_shap:
        st.markdown(f"#### Top risk drivers — {quarter_label(latest_year, latest_period)}")

        company_shap = get_shap_for_company(shap_df, selected_firm)

        if not company_shap.empty and "top_features_json" in company_shap.columns:
            # Use the most recent row's top_features_json
            shap_row = company_shap.iloc[-1]
            features = parse_top_features_json(shap_row["top_features_json"])

            if features:
                # Find max absolute value for bar scaling
                max_abs = max(abs(f["shap_value"]) for f in features) or 1.0

                for feat in features:
                    val = feat["shap_value"]
                    pct = min(abs(val) / max_abs * 100, 100)
                    color = shap_color(val)
                    sign = "+" if val > 0 else ""

                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:13px">
                            <div style="width:160px;color:#73726c;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{feat['feature']}</div>
                            <div style="flex:1;height:8px;background:#f0efea;border-radius:4px;overflow:hidden">
                                <div style="height:100%;width:{pct:.0f}%;background:{color};border-radius:4px"></div>
                            </div>
                            <div style="width:50px;text-align:right;color:{color};font-size:12px">{sign}{val:.3f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.caption("🟢 Green = reduces risk · 🔴 Red = increases risk")
            else:
                st.info("No SHAP explanations available for this quarter.")
        else:
            st.info(
                "SHAP data not available for this company. "
                "SHAP values are computed on the test set (2022-2023)."
            )

    # ── Financial snapshot ───────────────────────────────────────────
    with col_snapshot:
        st.markdown(f"#### Financial snapshot — {quarter_label(latest_year, latest_period)}")

        snapshot_fields = [
            ("Net income", "net_income", fmt_large_number),
            ("Total assets", "total_assets", fmt_large_number),
            ("Total liabilities", "total_liabilities", fmt_large_number),
            ("Stockholders equity", "StockholdersEquity", fmt_large_number),
            ("Cash & equivalents", "CashAndCashEquivalentsAtCarryingValue", fmt_large_number),
            ("Operating cash flow", "NetCashProvidedByUsedInOperatingActivities", fmt_large_number),
            ("Retained earnings", "RetainedEarningsAccumulatedDeficit", fmt_large_number),
        ]

        for label, col, formatter in snapshot_fields:
            if col in latest.index:
                val = latest[col]
                formatted = formatter(val) if val != 0 else "—"
                st.markdown(
                    f"""<div style="display:flex;justify-content:space-between;padding:5px 0;
                    border-bottom:0.5px solid rgba(0,0,0,0.07);font-size:13px">
                        <span style="color:#73726c">{label}</span>
                        <span style="font-weight:500">{formatted}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Model info footer
        st.markdown(
            f"""<div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px">
                <span style="color:#73726c">Model version</span>
                <span style="color:#9c9a92">v1</span>
            </div>""",
            unsafe_allow_html=True,
        )