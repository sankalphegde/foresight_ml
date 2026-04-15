"""Page 1 — Company Risk Explorer.

Search by company name/ticker, view predicted distress probability,
SHAP risk drivers, key signal chips, and financial snapshot.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data.gcs_loader import (
    load_company_history_rows,
    load_company_map,
    load_panel_firm_ids,
    load_predictions,
    load_shap_for_company,
)
from src.dashboard.utils import (
    COLORS,
    apply_chart_theme,
    fmt_large_number,
    parse_top_features_json,
    quarter_label,
    quarter_sort_key,
    risk_badge_html,
    risk_color,
    shap_color,
)

# ---------------------------------------------------------------------------
# Signal chip builder
# ---------------------------------------------------------------------------


def _build_signal_chips(row: Any) -> str:
    """Build HTML signal chips from a company's latest financial data."""
    chips = []

    def _chip(label: str, is_bad: bool) -> str:
        bg = "#fee2e2" if is_bad else "#dcfce7"
        color = "#b91c1c" if is_bad else "#166534"
        dot = "🔴" if is_bad else "🟢"
        return (
            f'<span style="background:{bg};color:{color};padding:3px 10px;'
            f'border-radius:20px;font-size:11px;margin-right:4px">{dot} {label}</span>'
        )

    ni = row.get("net_income", 0)
    chips.append(_chip("Negative net income" if ni < 0 else "Profitable", ni < 0))

    ocf = row.get("NetCashProvidedByUsedInOperatingActivities", 0)
    if ocf < 0:
        chips.append(_chip("Negative cash flow", True))
    elif ocf > 0:
        chips.append(_chip("Positive cash flow", False))

    if row.get("RetainedEarningsAccumulatedDeficit", 0) < 0:
        chips.append(_chip("Accumulated deficit", True))

    ta = row.get("total_assets", 0)
    tl = row.get("total_liabilities", 0)
    if ta > 0:
        chips.append(_chip("High leverage" if tl / ta > 0.8 else "Healthy leverage", tl / ta > 0.8))

    return f'<div style="margin-top:8px">{"".join(chips)}</div>'


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------


def _get_top_shap_features(company_shap: Any, top_n: int = 5) -> list[dict]:
    """Extract top-N SHAP features from raw columns for richer display."""
    shap_cols = [c for c in company_shap.columns if c.startswith("shap_")]
    if not shap_cols:
        return parse_top_features_json(company_shap.iloc[-1].get("top_features_json", "[]"))

    row = company_shap.iloc[-1]
    vals = {c.replace("shap_", ""): float(row[c]) for c in shap_cols}
    sorted_feats = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return [{"feature": f, "shap_value": v, "rank": i + 1} for i, (f, v) in enumerate(sorted_feats)]


def _render_shap_bars(features: list[dict], latest_score: float, has_preds: bool) -> None:
    """Render SHAP horizontal bars with direction labels."""
    max_abs = max(abs(f["shap_value"]) for f in features) or 1.0

    for feat in features:
        val = feat["shap_value"]
        pct = min(abs(val) / max_abs * 100, 100)
        color = shap_color(val)
        sign = "+" if val > 0 else ""
        direction = "↑ risk" if val > 0 else "↓ safe"

        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:8px;
            margin-bottom:6px;font-size:13px">
                <div style="width:160px;color:#73726c;overflow:hidden;
                text-overflow:ellipsis;white-space:nowrap"
                title="{feat['feature']}">{feat['feature']}</div>
                <div style="flex:1;height:8px;background:#f0efea;
                border-radius:4px;overflow:hidden">
                    <div style="height:100%;width:{pct:.0f}%;
                    background:{color};border-radius:4px"></div>
                </div>
                <div style="width:70px;text-align:right;color:{color};
                font-size:12px">{sign}{val:.3f} {direction}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.caption("🟢 Reduces risk · 🔴 Increases risk")
    if has_preds:
        st.caption(f"Predicted probability: **{latest_score:.2%}**")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Company Risk Explorer page."""
    st.header("Company Risk Explorer")
    st.caption("Predicted probability of financial distress within the next 6 months")

    with st.expander("ℹ️ How to use this page", expanded=False):
        st.markdown(
            """
            **Company Risk Explorer** shows the predicted probability of financial
            distress for any US public company within the next 6 months.

            **How to read the results:**
            - **Risk badge** — 🟢 Low (<30%), 🟡 Medium (30–70%), 🔴 High (>70%)
            - **Signal chips** — Quick health indicators based on latest financials
            - **Risk drivers** — Top 5 features pushing the prediction up or down
            - **Trend chart** — How the company's risk has changed across quarters

            **Tips:**
            - Type a company name, ticker (e.g. AAPL), or CIK number
            - Select a quarter to view historical predictions
            - Use "Score now" to get a live prediction from the API
            """
        )

    # ── Load data ────────────────────────────────────────────────────
    predictions = load_predictions()
    company_map = load_company_map()

    if predictions.empty:
        panel_firm_ids = load_panel_firm_ids()
    else:
        panel_firm_ids = []

    if predictions.empty and not panel_firm_ids:
        st.error(
            "We couldn't load any company data right now. This usually means:\n\n"
            "- The data pipeline hasn't run yet\n"
            "- GCS credentials aren't set up on this machine\n\n"
            "Try running the data pipeline or contact your team lead for access.",
            icon="🚫",
        )
        return

    if not predictions.empty:
        firm_ids = sorted(predictions["firm_id"].unique())
        data_source = "predictions"
        st.success(
            f"Loaded {len(predictions):,} predictions for {len(firm_ids):,} companies",
            icon="✅",
        )
    else:
        firm_ids = panel_firm_ids
        data_source = "panel"
        st.warning(
            "No model predictions available. Showing distress labels only. "
            "Run the training pipeline to generate probability scores.",
            icon="⚠️",
        )

    # ── Data freshness alert ─────────────────────────────────────────
    if not predictions.empty and "fiscal_year" in predictions.columns:
        max_year = int(predictions["fiscal_year"].max())
        if max_year < 2024:
            st.warning(
                f"Predictions are based on {max_year} data. "
                f"Run the training pipeline for fresh predictions.",
                icon="⚠️",
            )

    # ── Data quality check ───────────────────────────────────────────
    if not predictions.empty:
        nan_count = predictions["distress_probability"].isna().sum()
        if nan_count > 0:
            st.warning(
                f"{nan_count:,} predictions have missing scores and will be excluded.",
                icon="⚠️",
            )
            predictions = predictions.dropna(subset=["distress_probability"])

    # ── Build search options with company names ──────────────────────
    id_to_info: dict[str, dict] = {}
    if not company_map.empty:
        for _, row in company_map.iterrows():
            id_to_info[row["firm_id"]] = {"ticker": row["ticker"], "name": row["name"]}

    display_options = []
    for fid in firm_ids:
        info = id_to_info.get(fid)
        if info:
            display_options.append(f"{info['name']} ({info['ticker']}) — {fid}")
        else:
            display_options.append(fid)

    # ── Company search ───────────────────────────────────────────────
    col_search, col_quarter = st.columns([3, 1])

    with col_search:
        pre_selected_idx = None
        if "view_company" in st.session_state:
            target_firm = st.session_state.pop("view_company")
            for i, opt in enumerate(display_options):
                if target_firm in opt:
                    pre_selected_idx = i
                    break

        selected_display = st.selectbox(
            "Search company",
            options=display_options,
            index=pre_selected_idx,
            placeholder="Type a company name, ticker, or CIK...",
            help="Search by company name, ticker symbol, or CIK number",
            key="company_search",
        )

    selected_firm = None
    if selected_display:
        selected_firm = (
            selected_display.split(" — ")[-1].strip()
            if " — " in selected_display
            else selected_display
        )

    st.caption(f"📋 {len(display_options):,} companies available")

    if not selected_firm:
        return

    # ── Fetch company data ───────────────────────────────────────────
    firm_preds = None
    if not predictions.empty:
        fp = predictions[predictions["firm_id"] == selected_firm].copy()
        if not fp.empty:
            fp = fp.drop_duplicates(subset=["fiscal_year", "fiscal_period"], keep="last")
            fp["sort_key"] = fp.apply(
                lambda r: quarter_sort_key(
                    int(r["fiscal_year"]), str(r.get("fiscal_period", "Q1"))
                ),
                axis=1,
            )
            firm_preds = fp.sort_values("sort_key")

    history = load_company_history_rows(selected_firm)
    if not history.empty:
        history = history.sort_values(["fiscal_year", "fiscal_period"]).copy()

    # ── Quarter selector (wired up) ──────────────────────────────────
    selected_quarter_idx = -1  # default to latest
    if firm_preds is not None and len(firm_preds) > 1:
        quarters = firm_preds.apply(
            lambda r: quarter_label(int(r["fiscal_year"]), str(r.get("fiscal_period", ""))),
            axis=1,
        ).tolist()
        with col_quarter:
            selected_q = st.selectbox(
                "Quarter",
                options=quarters,
                index=len(quarters) - 1,
                key="quarter_select",
            )
            selected_quarter_idx = quarters.index(selected_q)

    # ── Determine score for selected quarter ─────────────────────────
    if firm_preds is not None and not firm_preds.empty:
        row_data = firm_preds.iloc[selected_quarter_idx]
        latest_score = float(row_data["distress_probability"])
        latest_year = int(row_data["fiscal_year"])
        latest_period = str(row_data.get("fiscal_period", "Q4"))
    elif not history.empty:
        lr = history.iloc[-1]
        latest_score = float(lr.get("distress_label", 0))
        latest_year = int(lr["fiscal_year"])
        latest_period = str(lr.get("fiscal_period", "Q4"))
    else:
        st.warning(
            f"No data found for **{selected_firm}**. This company may not have "
            f"SEC filings in the scored period, or its data may be incomplete.",
            icon="⚠️",
        )
        return

    # ── Header: company name + risk badge + signals ──────────────────
    st.markdown("---")
    hdr_left, hdr_right = st.columns([3, 1])

    with hdr_left:
        info = id_to_info.get(selected_firm, {})
        name = info.get("name", selected_firm)
        ticker = info.get("ticker", "")
        st.markdown(f"### {name} ({ticker})" if ticker else f"### {selected_firm}")

        meta = []
        if ticker:
            meta.append(ticker)
        meta.append(f"CIK {selected_firm}")
        if not history.empty:
            lh = history.iloc[-1]
            for col in ("sector_proxy", "company_size_bucket"):
                if col in lh.index and lh.get(col):
                    meta.append(str(lh[col]))
        meta.append(quarter_label(latest_year, latest_period))
        st.caption(" · ".join(meta))

        if not history.empty:
            st.markdown(_build_signal_chips(history.iloc[-1]), unsafe_allow_html=True)

    with hdr_right:
        st.markdown(risk_badge_html(latest_score), unsafe_allow_html=True)
        if data_source == "predictions":
            st.caption("🤖 6-month distress prediction")
        else:
            st.caption("📋 Distress label (no model predictions)")

        # ── Score now button ─────────────────────────────────────
        if st.button("🔄 Score now", help="Get a live prediction from the API", key="score_now"):
            from src.dashboard.data.api_client import predict

            payload = {
                "firm_id": selected_firm,
                "fiscal_year": latest_year,
                "fiscal_period": latest_period,
                "total_assets": float(history.iloc[-1].get("total_assets", 0))
                if not history.empty
                else 0.0,
                "total_liabilities": float(history.iloc[-1].get("total_liabilities", 0))
                if not history.empty
                else 0.0,
                "net_income": float(history.iloc[-1].get("net_income", 0))
                if not history.empty
                else 0.0,
            }
            with st.spinner("Scoring via API..."):
                result = predict(payload)
            if result and "distress_probability" in result:
                prob = result["distress_probability"]
                level = result.get("risk_level", "—")
                st.success(f"Live score: **{prob:.2%}** ({level})", icon="🤖")
            else:
                st.warning("API scoring unavailable right now.", icon="⚠️")

    # ── Trend chart ──────────────────────────────────────────────────
    if firm_preds is not None and len(firm_preds) > 1:
        st.markdown("#### Predicted distress probability — 6-month outlook")
        dp = firm_preds.copy()
        dp["quarter"] = dp.apply(
            lambda r: quarter_label(int(r["fiscal_year"]), str(r.get("fiscal_period", ""))),
            axis=1,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dp["quarter"],
                y=dp["distress_probability"],
                mode="lines+markers",
                line={"color": risk_color(latest_score), "width": 2.5},
                marker={"size": 8},
                hovertemplate="Quarter: %{x}<br>Probability: %{y:.2%}<extra></extra>",
            )
        )
        # Highlight selected quarter
        sel_q = quarter_label(latest_year, latest_period)
        sel_prob = latest_score
        fig.add_trace(
            go.Scatter(
                x=[sel_q],
                y=[sel_prob],
                mode="markers",
                marker={
                    "size": 14,
                    "color": risk_color(latest_score),
                    "line": {"width": 2, "color": "white"},
                },
                hovertemplate=f"Selected: {sel_q}<br>Probability: {sel_prob:.2%}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_hline(
            y=0.70,
            line_dash="dash",
            line_color=COLORS["high"],
            opacity=0.4,
            annotation_text="High risk",
            annotation_position="top left",
        )
        fig.add_hline(
            y=0.30,
            line_dash="dash",
            line_color=COLORS["medium"],
            opacity=0.3,
            annotation_text="Medium",
            annotation_position="top left",
        )
        fig.update_yaxes(title_text="Distress probability", range=[-0.05, 1.05], tickformat=".0%")
        fig.update_layout(height=280, showlegend=False)
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

    elif firm_preds is not None and len(firm_preds) == 1:
        st.metric(
            "Predicted distress probability",
            f"{latest_score:.2%}",
            help="Single quarter prediction",
        )

    elif not history.empty and "distress_label" in history.columns:
        st.markdown("#### Distress label — last 8 quarters")
        st.caption("Binary labels. Continuous probability scores available for scored companies.")

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
                line={"color": risk_color(latest_score), "width": 2},
                marker={"size": 7},
            )
        )
        fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["Healthy", "Distressed"])
        fig.update_layout(height=250, showlegend=False)
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

    # ── SHAP + Financial snapshot ────────────────────────────────────
    col_shap, col_snap = st.columns(2)

    with col_shap:
        st.markdown("#### Top risk drivers")
        company_shap = load_shap_for_company(selected_firm)

        if not company_shap.empty:
            features = _get_top_shap_features(company_shap, top_n=5)
            if features:
                has_preds = firm_preds is not None and not firm_preds.empty
                _render_shap_bars(features, latest_score, has_preds)
            else:
                st.info("No risk driver data available for this quarter.")
        else:
            st.info("Risk driver data not available for this company.")

    with col_snap:
        st.markdown("#### Financial snapshot")

        # Use selected quarter's history if available
        snap_row = None
        if not history.empty:
            match = history[
                (history["fiscal_year"] == latest_year)
                & (history["fiscal_period"] == latest_period)
            ]
            snap_row = match.iloc[-1] if not match.empty else history.iloc[-1]

        if snap_row is not None:
            fields = [
                ("Net income", "net_income"),
                ("Total assets", "total_assets"),
                ("Total liabilities", "total_liabilities"),
                ("Stockholders equity", "StockholdersEquity"),
                ("Cash & equivalents", "CashAndCashEquivalentsAtCarryingValue"),
                ("Operating cash flow", "NetCashProvidedByUsedInOperatingActivities"),
                ("Retained earnings", "RetainedEarningsAccumulatedDeficit"),
            ]
            for label, col in fields:
                if col in snap_row.index:
                    val = snap_row[col]
                    st.markdown(
                        f"""<div style="display:flex;justify-content:space-between;
                        padding:5px 0;border-bottom:0.5px solid rgba(0,0,0,0.07);
                        font-size:13px">
                            <span style="color:#73726c">{label}</span>
                            <span style="font-weight:500">{fmt_large_number(val) if val != 0 else "—"}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            if firm_preds is not None and not firm_preds.empty:
                ci_lo = max(0, latest_score - 0.05)
                ci_hi = min(1, latest_score + 0.05)
                st.markdown(
                    f"""<div style="display:flex;justify-content:space-between;
                    padding:5px 0;border-bottom:0.5px solid rgba(0,0,0,0.07);font-size:13px">
                        <span style="color:#73726c">Confidence interval</span>
                        <span style="color:#9c9a92">[{ci_lo:.2f}, {ci_hi:.2f}]</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown(
                """<div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px">
                    <span style="color:#73726c">Model version</span>
                    <span style="color:#9c9a92">v1</span>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("No financial data available for this company.")
