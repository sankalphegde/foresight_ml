"""Page 2 — High-Risk Watchlist.

Filterable table of companies ranked by predicted distress probability,
with company names, sector breakdown chart, filters, and CSV export.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data.gcs_loader import (
    load_company_map,
    load_labeled_panel,
    load_predictions,
)
from src.dashboard.utils import risk_emoji


# ---------------------------------------------------------------------------
# Watchlist builder
# ---------------------------------------------------------------------------


def _build_watchlist(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-company watchlist with risk scores and signals."""
    if predictions.empty:
        return pd.DataFrame()

    predictions = predictions.sort_values(["firm_id", "fiscal_year", "fiscal_period"])
    latest = predictions.groupby("firm_id").last().reset_index()

    # Quarter-over-quarter trend
    prev = predictions.groupby("firm_id").nth(-2).reset_index()
    if not prev.empty and "distress_probability" in prev.columns:
        prev_col = prev[["firm_id", "distress_probability"]].rename(
            columns={"distress_probability": "prev_prob"}
        )
        latest = latest.merge(prev_col, on="firm_id", how="left")
    else:
        latest["prev_prob"] = None

    # Merge financial signals from panel
    if not panel.empty:
        panel_latest = (
            panel.sort_values(["firm_id", "fiscal_year", "fiscal_period"])
            .groupby("firm_id").last().reset_index()
        )
        signal_cols = [
            "sector_proxy", "company_size_bucket", "net_income",
            "NetCashProvidedByUsedInOperatingActivities",
            "RetainedEarningsAccumulatedDeficit", "total_liabilities", "total_assets",
        ]
        merge_cols = ["firm_id"] + [c for c in signal_cols if c in panel_latest.columns]
        latest = latest.merge(panel_latest[merge_cols], on="firm_id", how="left")

    # Build rows
    rows = []
    for _, r in latest.iterrows():
        prob = float(r.get("distress_probability", 0))

        signals = []
        if pd.notna(r.get("net_income")) and r["net_income"] < 0:
            signals.append("Neg. income")
        if pd.notna(r.get("NetCashProvidedByUsedInOperatingActivities")) and r["NetCashProvidedByUsedInOperatingActivities"] < 0:
            signals.append("Neg. cash flow")
        if pd.notna(r.get("RetainedEarningsAccumulatedDeficit")) and r["RetainedEarningsAccumulatedDeficit"] < 0:
            signals.append("Retained earnings ↓")
        if pd.notna(r.get("total_assets")) and r.get("total_assets", 0) > 0 and pd.notna(r.get("total_liabilities")) and r["total_liabilities"] / r["total_assets"] > 0.8:
            signals.append("High leverage")

        prev_val = r.get("prev_prob")
        change = prob - float(prev_val) if pd.notna(prev_val) else 0.0

        rows.append({
            "firm_id": r["firm_id"],
            "sector": r.get("sector_proxy", "—"),
            "size": r.get("company_size_bucket", "—"),
            "risk_score": prob,
            "change": change,
            "signals": " · ".join(signals) if signals else "—",
            "quarter": f"{r.get('fiscal_period', '')} {int(r.get('fiscal_year', 0))}",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the High-Risk Watchlist page."""
    st.header("🔴 High-Risk Watchlist")
    st.caption("Companies most likely to experience financial distress within the next 6 months")
    with st.expander("ℹ️ How to use this page", expanded=False):
        st.markdown(
            """
            **High-Risk Watchlist** ranks all scored companies by predicted
            distress probability. Use it to identify which companies need attention.

            **How to read the table:**
            - **Risk Score** — Predicted probability of distress in next 6 months
            - **vs Last Qtr** — 🔴 risk increased, 🟢 risk decreased since last quarter
            - **Active Distress Signals** — Financial warning signs from latest filing

            **Tips:**
            - Use the threshold slider to focus on high-risk companies only
            - Export to CSV for offline analysis or sharing with your team
            - Sector/size filters help narrow down specific segments
            """
        )

    predictions = load_predictions()
    panel = load_labeled_panel()
    company_map = load_company_map()

    if predictions.empty:
        st.warning(
            "No model predictions available. Run the training pipeline to generate scores.",
            icon="⚠️",
        )
        return

    watchlist = _build_watchlist(predictions, panel)
    if watchlist.empty:
        st.warning("Could not build watchlist.")
        return

    # ── Filters ──────────────────────────────────────────────────────
    col_thresh, col_sector, col_size, col_export = st.columns([2, 1.5, 1.5, 1])

    with col_thresh:
        threshold = st.slider(
            "Minimum risk score", min_value=0.0, max_value=1.0,
            value=0.5, step=0.05,
            help="Show companies with predicted distress probability ≥ this value. "
                 "High risk ≥ 0.70, Medium ≥ 0.30",
        )

    with col_sector:
        sectors = ["All sectors"] + sorted(watchlist["sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("Sector", options=sectors, help="Filter by industry sector")

    with col_size:
        sizes = ["All sizes"] + sorted(watchlist["size"].dropna().unique().tolist())
        selected_size = st.selectbox("Size", options=sizes, help="Filter by company size bucket")

    # Apply filters
    filtered = watchlist[watchlist["risk_score"] >= threshold].copy()
    if selected_sector != "All sectors":
        filtered = filtered[filtered["sector"] == selected_sector]
    if selected_size != "All sizes":
        filtered = filtered[filtered["size"] == selected_size]
    filtered = filtered.sort_values("risk_score", ascending=False)

    with col_export:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Export CSV",
            data=filtered.to_csv(index=False),
            file_name="watchlist_export.csv",
            mime="text/csv",
            help="Download filtered watchlist as CSV",
        )

    # ── Summary metrics ──────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Companies shown", f"{len(filtered):,}")
    m2.metric("🔴 High risk (≥0.70)", len(filtered[filtered["risk_score"] >= 0.70]))
    m3.metric("🟡 Medium (0.30–0.70)", len(filtered[(filtered["risk_score"] >= 0.30) & (filtered["risk_score"] < 0.70)]))
    m4.metric("Total scored", f"{len(watchlist):,}")

    # ── Sector breakdown chart ───────────────────────────────────────
    if not filtered.empty and "sector" in filtered.columns:
        sector_counts = filtered["sector"].value_counts()
        sector_counts = sector_counts[sector_counts.index != "—"]
        if len(sector_counts) >= 1:
            col_chart, _ = st.columns([1, 2])
            with col_chart:
                st.markdown("#### Risk by sector")
                palette = ["#b91c1c", "#d97706", "#3b7dd8", "#16a34a", "#9333ea", "#0891b2", "#be185d", "#4f46e5"]
                fig = go.Figure(data=[go.Pie(
                    labels=sector_counts.index.tolist(),
                    values=sector_counts.values.tolist(),
                    hole=0.5,
                    marker={"colors": palette[:len(sector_counts)]},
                    textinfo="label+value",
                    textfont={"size": 11},
                    hovertemplate="%{label}: %{value} companies<extra></extra>",
                )])
                fig.update_layout(
                    height=220, showlegend=False,
                    margin={"l": 0, "r": 0, "t": 10, "b": 10},
                    paper_bgcolor="white", plot_bgcolor="white",
                )
                st.plotly_chart(fig, width="stretch")

    # ── Add company names ────────────────────────────────────────────
    if not company_map.empty:
        name_map = dict(zip(company_map["firm_id"], company_map["name"]))
        ticker_map = dict(zip(company_map["firm_id"], company_map["ticker"]))
        filtered["company"] = filtered["firm_id"].map(name_map).fillna("—")
        filtered["ticker"] = filtered["firm_id"].map(ticker_map).fillna("—")
    else:
        filtered["company"] = "—"
        filtered["ticker"] = "—"

    # ── Watchlist table ──────────────────────────────────────────────
    if filtered.empty:
        st.info(f"No companies found with risk score ≥ {threshold:.0%}")
        if threshold > 0.9:
            st.info(
                "💡 Try lowering the threshold. Most companies have low distress probability, "
                "which is expected — only ~2-5% of companies experience financial distress."
            )
        return

    st.markdown(f"**Showing {len(filtered):,} companies** · Sorted by predicted distress probability")

    display = filtered.copy()
    display["risk"] = display["risk_score"].apply(lambda s: f"{risk_emoji(s)} {s:.2%}")
    display["trend"] = display["change"].apply(
        lambda c: f"🔴 +{c:.2%} ↑" if c > 0.01
        else (f"🟢 {c:.2%} ↓" if c < -0.01 else "— 0.00")
    )

    col_map = {
        "company": "Company", "ticker": "Ticker", "firm_id": "CIK",
        "sector": "Sector", "size": "Size", "risk": "Risk Score",
        "trend": "vs Last Qtr", "signals": "Active Distress Signals", "quarter": "Quarter",
    }
    st.dataframe(
        display[list(col_map.keys())].rename(columns=col_map),
        use_container_width=True,
        hide_index=True,
        height=min(len(display) * 38 + 40, 600),
    )

    st.caption(
        f"Predictions from XGBoost model (test set 2022–2023). "
        f"Threshold: {threshold:.0%} · {len(filtered):,} of {len(watchlist):,} companies."
    )