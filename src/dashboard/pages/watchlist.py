"""Page 2 — High-Risk Watchlist.

Filterable table of companies ranked by predicted distress probability,
with company names, sector breakdown chart, filters, and CSV export.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.data.gcs_loader import (
    load_company_map,
    load_labeled_panel,
    load_predictions,
)
from src.dashboard.utils import risk_emoji

# ---------------------------------------------------------------------------
# Column name mapping — handles both old and new panel schemas
# ---------------------------------------------------------------------------
_COL_MAP = {
    "operating_cash_flow": "operating_cash_flow",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "retained_earnings": "retained_earnings",
    "RetainedEarningsAccumulatedDeficit": "retained_earnings",
    "total_equity": "total_equity",
    "StockholdersEquity": "total_equity",
}


def _get_col(row: pd.Series, *candidates: str, default: float = 0) -> float:
    """Get the first available column value from a row."""
    for c in candidates:
        val = row.get(c)
        if pd.notna(val):
            return float(val)
    return default


# ---------------------------------------------------------------------------
# Watchlist builder
# ---------------------------------------------------------------------------


def _build_watchlist(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-company watchlist with risk scores and signals."""
    if predictions.empty:
        return pd.DataFrame()

    predictions = predictions.sort_values(["firm_id", "fiscal_year", "fiscal_period"])
    latest = predictions.drop_duplicates(subset=["firm_id"], keep="last").copy()

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
        panel_latest = panel.sort_values(
            ["firm_id", "fiscal_year", "fiscal_period"]
        ).drop_duplicates(subset=["firm_id"], keep="last")
        signal_cols = [
            "net_income",
            "operating_cash_flow",
            "NetCashProvidedByUsedInOperatingActivities",
            "retained_earnings",
            "RetainedEarningsAccumulatedDeficit",
            "total_liabilities",
            "total_assets",
        ]
        merge_cols = ["firm_id"] + [c for c in signal_cols if c in panel_latest.columns]
        latest = latest.merge(panel_latest[merge_cols], on="firm_id", how="left")

    # Build rows
    rows = []
    for _, r in latest.iterrows():
        prob = float(r.get("distress_probability", 0))

        signals = []
        ni = _get_col(r, "net_income")
        if ni < 0:
            signals.append("Neg. income")

        ocf = _get_col(r, "operating_cash_flow", "NetCashProvidedByUsedInOperatingActivities")
        if ocf < 0:
            signals.append("Neg. cash flow")

        re = _get_col(r, "retained_earnings", "RetainedEarningsAccumulatedDeficit")
        if re < 0:
            signals.append("Retained earnings deficit")

        ta = _get_col(r, "total_assets")
        tl = _get_col(r, "total_liabilities")
        if ta > 0 and tl / ta > 0.8:
            signals.append("High leverage")

        prev_val = r.get("prev_prob")
        change = prob - float(prev_val) if pd.notna(prev_val) else 0.0

        rows.append(
            {
                "firm_id": r["firm_id"],
                "risk_score": prob,
                "change": change,
                "signals": " · ".join(signals) if signals else "—",
                "quarter": f"{r.get('fiscal_period', '')} {int(r.get('fiscal_year', 0))}",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the High-Risk Watchlist page."""
    st.header("High-Risk Watchlist")
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
            - Select a company and click "View details" to see the full risk analysis
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
    col_thresh, col_spacer, col_export = st.columns([2, 2, 1])

    with col_thresh:
        threshold = st.slider(
            "Minimum risk score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Show companies with predicted distress probability above this value. "
            "High risk ≥ 0.70, Medium ≥ 0.30",
        )

    # Apply filters
    filtered = watchlist[watchlist["risk_score"] >= threshold].copy()
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
    m1.metric(
        "Companies shown", f"{len(filtered):,}", help="Number of companies matching current filter"
    )
    m2.metric(
        "🔴 High risk (≥0.70)",
        len(filtered[filtered["risk_score"] >= 0.70]),
        help="Companies with >70% distress probability",
    )
    m3.metric(
        "🟡 Medium (0.30–0.70)",
        len(filtered[(filtered["risk_score"] >= 0.30) & (filtered["risk_score"] < 0.70)]),
        help="Companies with 30-70% distress probability",
    )
    m4.metric("Total scored", f"{len(watchlist):,}", help="All companies scored by the model")

    # ── Add company names ────────────────────────────────────────────
    if not company_map.empty:
        name_map = dict(zip(company_map["firm_id"], company_map["name"], strict=False))
        ticker_map = dict(zip(company_map["firm_id"], company_map["ticker"], strict=False))
        filtered["company"] = filtered["firm_id"].map(name_map).fillna("—")
        filtered["ticker"] = filtered["firm_id"].map(ticker_map).fillna("—")
    else:
        filtered["company"] = "—"
        filtered["ticker"] = "—"

    # ── Watchlist table ──────────────────────────────────────────────
    if filtered.empty:
        st.info(f"No companies found with risk score above {threshold:.0%}")
        if threshold > 0.9:
            st.info(
                "Try lowering the threshold. Most companies have low distress probability, "
                "which is expected — only 2-5% of companies experience financial distress."
            )
        return

    # ── View company selector (above table) ──────────────────────────
    view_col, info_col = st.columns([2, 3])
    with view_col:
        view_options = []
        for _, r in filtered.iterrows():
            label = f"{r['company']} ({r['ticker']})" if r["company"] != "—" else r["firm_id"]
            view_options.append(label)

        selected_view = st.selectbox(
            "View company details",
            options=view_options,
            index=None,
            placeholder="Select a company to view full risk analysis...",
            help="Select a company, then switch to Risk Analysis in the sidebar",
            key="watchlist_view",
        )

    if selected_view:
        idx = view_options.index(selected_view)
        firm_id = filtered.iloc[idx]["firm_id"]
        st.session_state["view_company"] = firm_id
        with info_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                "Company selected. Click **Risk Analysis** in the sidebar to view details.",
                icon="👈",
            )

    st.markdown(
        f"**Showing {len(filtered):,} companies** · Sorted by predicted distress probability"
    )

    display = filtered.copy()
    display["risk"] = display["risk_score"].apply(lambda s: f"{risk_emoji(s)} {s:.2%}")
    display["trend"] = display["change"].apply(
        lambda c: f"🔴 +{c:.2%} ↑" if c > 0.01 else (f"🟢 {c:.2%} ↓" if c < -0.01 else "— 0.00")
    )

    col_map = {
        "company": "Company",
        "ticker": "Ticker",
        "firm_id": "CIK",
        "risk": "Risk Score",
        "trend": "vs Last Qtr",
        "signals": "Distress Signals",
        "quarter": "Quarter",
    }
    st.dataframe(
        display[list(col_map.keys())].rename(columns=col_map),
        use_container_width=True,
        hide_index=True,
        height=min(len(display) * 38 + 40, 600),
    )

    st.caption(f"Threshold: {threshold:.0%} · {len(filtered):,} of {len(watchlist):,} companies")
