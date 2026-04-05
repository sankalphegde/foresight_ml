"""Page 1 — Company Risk Explorer.

Search by company name/ticker, view predicted distress probability,
SHAP risk drivers, key signal chips, and financial snapshot.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data.gcs_loader import (
    get_company_history,
    get_shap_for_company,
    load_company_map,
    load_labeled_panel,
    load_predictions,
    load_shap_values,
)
from src.dashboard.utils import (
    apply_chart_theme,
    fmt_large_number,
    parse_top_features_json,
    quarter_label,
    quarter_sort_key,
    risk_badge_html,
    risk_color,
    shap_color,
    COLORS,
)


# ---------------------------------------------------------------------------
# Signal chip builder
# ---------------------------------------------------------------------------


def _build_signal_chips(row: object) -> str:
    """Build HTML signal chips from a company's latest financial data."""
    chips = []

    def _chip(label: str, is_bad: bool) -> str:
        if is_bad:
            return (
                f'<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;'
                f'border-radius:20px;font-size:11px;margin-right:4px">🔴 {label}</span>'
            )
        return (
            f'<span style="background:#dcfce7;color:#166534;padding:3px 10px;'
            f'border-radius:20px;font-size:11px;margin-right:4px">🟢 {label}</span>'
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
# SHAP bar builder
# ---------------------------------------------------------------------------


def _get_top_shap_features(company_shap: object, top_n: int = 5) -> list[dict]:
    """Extract top-N SHAP features from raw columns for richer display."""
    shap_cols = [c for c in company_shap.columns if c.startswith("shap_")]
    if not shap_cols:
        return parse_top_features_json(
            company_shap.iloc[-1].get("top_features_json", "[]")
        )

    row = company_shap.iloc[-1]
    vals = {c.replace("shap_", ""): float(row[c]) for c in shap_cols}
    sorted_feats = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return [
        {"feature": f, "shap_value": v, "rank": i + 1}
        for i, (f, v) in enumerate(sorted_feats)
    ]


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

    st.caption("🟢 Green = reduces risk · 🔴 Red = increases risk")
    if has_preds:
        st.caption(f"Predicted probability: **{latest_score:.2%}**")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Company Risk Explorer page."""
    st.header("🔍 Company Risk Explorer")
    st.caption("Probability of financial distress within the next 6 months (2 quarters)")

    # ── Load data ────────────────────────────────────────────────────
    predictions = load_predictions()
    panel = load_labeled_panel()
    shap_df = load_shap_values()
    company_map = load_company_map()

    if predictions.empty and panel.empty:
        st.error(
            "No data available. Ensure GCS credentials are configured "
            "and the data pipeline has run at least once."
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
        firm_ids = sorted(panel["firm_id"].unique())
        data_source = "panel"
        st.warning(
            "No model predictions available. Showing distress labels only. "
            "Run the training pipeline to generate probability scores.",
            icon="⚠️",
        )

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
        selected_display = st.selectbox(
            "Search company",
            options=display_options,
            index=0,
            help="Type a company name, ticker symbol, or CIK number to search",
        )

    selected_firm = (
        selected_display.split(" — ")[-1].strip()
        if " — " in selected_display
        else selected_display
    )

    st.caption(
        f"📋 {len(display_options):,} companies available · "
        "Can't find a company? Only US public companies with SEC filings "
        "in the 2022–2023 test period are scored."
    )

    if not selected_firm:
        return

    # ── Loading spinner ──────────────────────────────────────────────
    loading = st.empty()
    loading.markdown(
        """<div style="display:flex;align-items:center;justify-content:center;
        padding:40px;gap:10px">
            <div style="width:20px;height:20px;border:2.5px solid #f0efea;
            border-top:2.5px solid #3b7dd8;border-radius:50%;
            animation:spin 0.8s linear infinite"></div>
            <span style="font-size:14px;color:#9c9a92">Analyzing company...</span>
        </div>
        <style>@keyframes spin { to { transform: rotate(360deg); } }</style>
        """,
        unsafe_allow_html=True,
    )

    # ── Fetch company data ───────────────────────────────────────────
    firm_preds = None
    if not predictions.empty:
        fp = predictions[predictions["firm_id"] == selected_firm].copy()
        if not fp.empty:
            fp["sort_key"] = fp.apply(
                lambda r: quarter_sort_key(
                    int(r["fiscal_year"]), str(r.get("fiscal_period", "Q1"))
                ),
                axis=1,
            )
            firm_preds = fp.sort_values("sort_key")

    history = get_company_history(panel, selected_firm)

    # Determine latest score
    if firm_preds is not None and not firm_preds.empty:
        lp = firm_preds.iloc[-1]
        latest_score = float(lp["distress_probability"])
        latest_year = int(lp["fiscal_year"])
        latest_period = str(lp.get("fiscal_period", "Q4"))
    elif not history.empty:
        lr = history.iloc[-1]
        latest_score = float(lr.get("distress_label", 0))
        latest_year = int(lr["fiscal_year"])
        latest_period = str(lr.get("fiscal_period", "Q4"))
    else:
        loading.empty()
        st.warning(f"No data found for {selected_firm}")
        return

    with col_quarter:
        if firm_preds is not None and len(firm_preds) > 1:
            quarters = firm_preds.apply(
                lambda r: quarter_label(
                    int(r["fiscal_year"]), str(r.get("fiscal_period", ""))
                ),
                axis=1,
            ).tolist()
            st.selectbox("Quarter", options=quarters, index=len(quarters) - 1)

    loading.empty()

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

        # Signal chips
        if not history.empty:
            st.markdown(_build_signal_chips(history.iloc[-1]), unsafe_allow_html=True)

    with hdr_right:
        st.markdown(risk_badge_html(latest_score), unsafe_allow_html=True)
        if data_source == "predictions":
            st.caption("🤖 6-month distress prediction (XGBoost)")
        else:
            st.caption("📋 Distress label (no model predictions)")

    # ── Trend chart ──────────────────────────────────────────────────
    if firm_preds is not None and len(firm_preds) > 1:
        st.markdown("#### Predicted distress probability — 6-month outlook")
        dp = firm_preds.copy()
        dp["quarter"] = dp.apply(
            lambda r: quarter_label(int(r["fiscal_year"]), str(r.get("fiscal_period", ""))),
            axis=1,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dp["quarter"], y=dp["distress_probability"],
            mode="lines+markers",
            line={"color": risk_color(latest_score), "width": 2.5},
            marker={"size": 8},
            hovertemplate="Quarter: %{x}<br>Probability: %{y:.2%}<extra></extra>",
        ))
        fig.add_hline(y=0.70, line_dash="dash", line_color=COLORS["high"], opacity=0.4,
                      annotation_text="High risk (0.70)", annotation_position="top left")
        fig.add_hline(y=0.30, line_dash="dash", line_color=COLORS["medium"], opacity=0.3,
                      annotation_text="Medium (0.30)", annotation_position="top left")
        fig.update_yaxes(title_text="Distress probability", range=[-0.05, 1.05], tickformat=".0%")
        fig.update_layout(height=280, showlegend=False)
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

    elif firm_preds is not None and len(firm_preds) == 1:
        st.metric("Predicted distress probability", f"{latest_score:.2%}",
                  help="Single quarter prediction from XGBoost model")

    elif not history.empty and "distress_label" in history.columns:
        st.markdown("#### Distress label — last 8 quarters")
        st.caption("⚠️ Binary labels. Continuous probability scores available for test set (2022–2023).")

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
        fig.add_trace(go.Scatter(
            x=recent["quarter"], y=recent["distress_label"],
            mode="lines+markers",
            line={"color": risk_color(latest_score), "width": 2}, marker={"size": 7},
        ))
        fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["Healthy", "Distressed"])
        fig.update_layout(height=250, showlegend=False)
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

    # ── SHAP + Financial snapshot ────────────────────────────────────
    col_shap, col_snap = st.columns(2)

    with col_shap:
        st.markdown("#### Top risk drivers")
        company_shap = get_shap_for_company(shap_df, selected_firm)

        if not company_shap.empty:
            features = _get_top_shap_features(company_shap, top_n=5)
            if features:
                has_preds = firm_preds is not None and not firm_preds.empty
                _render_shap_bars(features, latest_score, has_preds)
            else:
                st.info("No SHAP explanations available for this quarter.")
        else:
            st.info(
                "SHAP data not available for this company. "
                "SHAP values are computed on the test set (2022–2023)."
            )

    with col_snap:
        st.markdown("#### Financial snapshot")
        if not history.empty:
            lh = history.iloc[-1]
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
                if col in lh.index:
                    val = lh[col]
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
                ci_lo, ci_hi = max(0, latest_score - 0.05), min(1, latest_score + 0.05)
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