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
from src.dashboard.utils import apply_chart_theme, risk_emoji, COLORS


def _build_watchlist(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Build a watchlist table with one row per company."""
    if predictions.empty:
        return pd.DataFrame()

    predictions = predictions.sort_values(["firm_id", "fiscal_year", "fiscal_period"])
    latest_preds = predictions.groupby("firm_id").last().reset_index()

    second_preds = predictions.groupby("firm_id").nth(-2).reset_index()
    if not second_preds.empty and "distress_probability" in second_preds.columns:
        prev = second_preds[["firm_id", "distress_probability"]].rename(
            columns={"distress_probability": "prev_prob"}
        )
        latest_preds = latest_preds.merge(prev, on="firm_id", how="left")
    else:
        latest_preds["prev_prob"] = None

    if not panel.empty:
        panel_sorted = panel.sort_values(["firm_id", "fiscal_year", "fiscal_period"])
        panel_latest = panel_sorted.groupby("firm_id").last().reset_index()

        merge_cols = ["firm_id"]
        optional = [
            "sector_proxy", "company_size_bucket", "net_income",
            "NetCashProvidedByUsedInOperatingActivities",
            "RetainedEarningsAccumulatedDeficit", "total_liabilities", "total_assets",
        ]
        for c in optional:
            if c in panel_latest.columns:
                merge_cols.append(c)

        latest_preds = latest_preds.merge(panel_latest[merge_cols], on="firm_id", how="left")

    rows = []
    for _, r in latest_preds.iterrows():
        prob = float(r.get("distress_probability", 0))

        signals = []
        if pd.notna(r.get("net_income")) and r["net_income"] < 0:
            signals.append("Neg. income")
        if (
            pd.notna(r.get("NetCashProvidedByUsedInOperatingActivities"))
            and r["NetCashProvidedByUsedInOperatingActivities"] < 0
        ):
            signals.append("Neg. cash flow")
        if (
            pd.notna(r.get("RetainedEarningsAccumulatedDeficit"))
            and r["RetainedEarningsAccumulatedDeficit"] < 0
        ):
            signals.append("Retained earnings ↓")
        if (
            pd.notna(r.get("total_assets")) and r.get("total_assets", 0) > 0
            and pd.notna(r.get("total_liabilities"))
            and r["total_liabilities"] / r["total_assets"] > 0.8
        ):
            signals.append("High leverage")

        prev = r.get("prev_prob")
        change = prob - float(prev) if pd.notna(prev) else 0.0

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


def render() -> None:
    """Render the High-Risk Watchlist page."""
    st.header("🔴 High-Risk Watchlist")
    st.caption("Companies most likely to experience financial distress within the next 6 months")

    predictions = load_predictions()
    panel = load_labeled_panel()
    company_map = load_company_map()

    if predictions.empty:
        st.warning("No model predictions available. Run batch inference first.")
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
            help="Show companies with predicted distress probability ≥ this value",
        )

    with col_sector:
        sectors = ["All sectors"] + sorted(watchlist["sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("Sector", options=sectors)

    with col_size:
        sizes = ["All sizes"] + sorted(watchlist["size"].dropna().unique().tolist())
        selected_size = st.selectbox("Size", options=sizes)

    filtered = watchlist[watchlist["risk_score"] >= threshold].copy()
    if selected_sector != "All sectors":
        filtered = filtered[filtered["sector"] == selected_sector]
    if selected_size != "All sizes":
        filtered = filtered[filtered["size"] == selected_size]

    filtered = filtered.sort_values("risk_score", ascending=False)

    with col_export:
        st.markdown("<br>", unsafe_allow_html=True)
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="📥 Export CSV", data=csv,
            file_name="watchlist_export.csv", mime="text/csv",
        )

    # ── Summary metrics ──────────────────────────────────────────────
    total = len(filtered)
    high_risk = len(filtered[filtered["risk_score"] >= 0.70])
    medium_risk = len(filtered[(filtered["risk_score"] >= 0.30) & (filtered["risk_score"] < 0.70)])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Companies shown", f"{total:,}")
    m2.metric("🔴 High risk (≥0.70)", high_risk)
    m3.metric("🟡 Medium (0.30–0.70)", medium_risk)
    m4.metric("Total scored", f"{len(watchlist):,}")

    # ── Sector breakdown chart ───────────────────────────────────────
    if not filtered.empty and "sector" in filtered.columns:
        sector_counts = filtered["sector"].value_counts()
        sector_counts = sector_counts[sector_counts.index != "—"]
        if len(sector_counts) >= 1:
            col_chart, col_space = st.columns([1, 2])
            with col_chart:
                st.markdown("#### Risk by sector")
                chart_colors = ["#b91c1c", "#d97706", "#3b7dd8", "#16a34a", "#9333ea", "#0891b2", "#be185d", "#4f46e5"]
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=sector_counts.index.tolist(),
                            values=sector_counts.values.tolist(),
                            hole=0.5,
                            marker={"colors": chart_colors[: len(sector_counts)]},
                            textinfo="label+value",
                            textfont={"size": 11},
                            hovertemplate="%{label}: %{value} companies<extra></extra>",
                        )
                    ]
                )
                fig.update_layout(
                    height=220,
                    margin={"l": 0, "r": 0, "t": 10, "b": 10},
                    showlegend=False,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig, width="stretch")

    # ── Add company names and tickers ────────────────────────────────
    if not company_map.empty:
        name_lookup = dict(zip(company_map["firm_id"], company_map["name"]))
        ticker_lookup = dict(zip(company_map["firm_id"], company_map["ticker"]))
        filtered["company"] = filtered["firm_id"].map(name_lookup).fillna("—")
        filtered["ticker"] = filtered["firm_id"].map(ticker_lookup).fillna("—")
    else:
        filtered["company"] = "—"
        filtered["ticker"] = "—"

    # ── Watchlist table ──────────────────────────────────────────────
    if filtered.empty:
        st.info(f"No companies found with risk score ≥ {threshold:.0%}")
        return

    st.markdown(f"**Showing {len(filtered):,} companies** · Sorted by predicted distress probability")

    display = filtered.copy()
    display["risk"] = display["risk_score"].apply(lambda s: f"{risk_emoji(s)} {s:.2%}")
    display["trend"] = display["change"].apply(
        lambda c: f"🔴 +{c:.2%} ↑" if c > 0.01
        else (f"🟢 {c:.2%} ↓" if c < -0.01 else "— 0.00")
    )

    show_cols = {
        "company": "Company",
        "ticker": "Ticker",
        "firm_id": "CIK",
        "sector": "Sector",
        "size": "Size",
        "risk": "Risk Score",
        "trend": "vs Last Qtr",
        "signals": "Active Distress Signals",
        "quarter": "Quarter",
    }
    display_df = display[list(show_cols.keys())].rename(columns=show_cols)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(len(display_df) * 38 + 40, 600),
    )

    st.caption(
        f"Predictions from XGBoost model (test set 2022–2023). "
        f"Threshold: {threshold:.0%} · {len(filtered):,} of {len(watchlist):,} companies shown."
    )