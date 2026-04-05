"""Page 1 — Company Risk Explorer.

Search by company name/ticker, view predicted distress probability trend,
SHAP risk drivers, key signal chips, and financial snapshot.
"""

from __future__ import annotations

import numpy as np
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
        st.error("No data available. Check GCS connection.")
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
        st.warning("No model predictions available. Showing distress labels only.", icon="⚠️")

    # Build display labels
    id_to_info = {}
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
            help="Type a company name, ticker, or CIK to search",
        )

    if " — " in selected_display:
        selected_firm = selected_display.split(" — ")[-1].strip()
    else:
        selected_firm = selected_display

    st.caption(
        f"📋 {len(display_options):,} companies available · "
        "Can't find a company? Only US public companies with SEC filings "
        "in the 2022–2023 test period are scored."
    )

    if not selected_firm:
        st.info("Select a company to view risk analysis.")
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

    # ── Get prediction data ──────────────────────────────────────────
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

    # ── Get financial data ───────────────────────────────────────────
    history = get_company_history(panel, selected_firm)

    if firm_preds is not None and not firm_preds.empty:
        latest_pred = firm_preds.iloc[-1]
        latest_score = float(latest_pred["distress_probability"])
        latest_year = int(latest_pred["fiscal_year"])
        latest_period = str(latest_pred.get("fiscal_period", "Q4"))
    elif not history.empty:
        latest_row = history.iloc[-1]
        latest_score = float(latest_row.get("distress_label", 0))
        latest_year = int(latest_row["fiscal_year"])
        latest_period = str(latest_row.get("fiscal_period", "Q4"))
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

    # ── Clear spinner ────────────────────────────────────────────────
    loading.empty()

    # ── Company header with risk badge ───────────────────────────────
    st.markdown("---")
    hdr_left, hdr_right = st.columns([3, 1])

    with hdr_left:
        info = id_to_info.get(selected_firm, {})
        company_name = info.get("name", selected_firm)
        ticker = info.get("ticker", "")
        display_title = f"{company_name} ({ticker})" if ticker else selected_firm
        st.markdown(f"### {display_title}")

        meta_parts = []
        if ticker:
            meta_parts.append(ticker)
        meta_parts.append(f"CIK {selected_firm}")
        if not history.empty:
            lh = history.iloc[-1]
            if "sector_proxy" in lh.index and lh.get("sector_proxy"):
                meta_parts.append(str(lh["sector_proxy"]))
            if "company_size_bucket" in lh.index and lh.get("company_size_bucket"):
                meta_parts.append(str(lh["company_size_bucket"]))
        meta_parts.append(quarter_label(latest_year, latest_period))
        st.caption(" · ".join(meta_parts))

        # ── Key signal chips ─────────────────────────────────────────
        if not history.empty:
            lh = history.iloc[-1]
            signals_html = []

            ni = lh.get("net_income", 0)
            if ni < 0:
                signals_html.append(
                    '<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;'
                    'border-radius:20px;font-size:11px;margin-right:4px">🔴 Negative net income</span>'
                )
            else:
                signals_html.append(
                    '<span style="background:#dcfce7;color:#166534;padding:3px 10px;'
                    'border-radius:20px;font-size:11px;margin-right:4px">🟢 Profitable</span>'
                )

            ocf = lh.get("NetCashProvidedByUsedInOperatingActivities", 0)
            if ocf < 0:
                signals_html.append(
                    '<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;'
                    'border-radius:20px;font-size:11px;margin-right:4px">🔴 Negative cash flow</span>'
                )
            elif ocf > 0:
                signals_html.append(
                    '<span style="background:#dcfce7;color:#166534;padding:3px 10px;'
                    'border-radius:20px;font-size:11px;margin-right:4px">🟢 Positive cash flow</span>'
                )

            re = lh.get("RetainedEarningsAccumulatedDeficit", 0)
            if re < 0:
                signals_html.append(
                    '<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;'
                    'border-radius:20px;font-size:11px;margin-right:4px">🔴 Accumulated deficit</span>'
                )

            ta = lh.get("total_assets", 0)
            tl = lh.get("total_liabilities", 0)
            if ta > 0 and tl / ta > 0.8:
                signals_html.append(
                    '<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;'
                    'border-radius:20px;font-size:11px;margin-right:4px">🔴 High leverage</span>'
                )
            elif ta > 0:
                signals_html.append(
                    '<span style="background:#dcfce7;color:#166534;padding:3px 10px;'
                    'border-radius:20px;font-size:11px;margin-right:4px">🟢 Healthy leverage</span>'
                )

            if signals_html:
                st.markdown(
                    f'<div style="margin-top:8px">{"".join(signals_html)}</div>',
                    unsafe_allow_html=True,
                )

    with hdr_right:
        st.markdown(risk_badge_html(latest_score), unsafe_allow_html=True)
        if data_source == "predictions":
            st.caption("🤖 6-month distress prediction (XGBoost)")
        else:
            st.caption("📋 Distress label (no model predictions)")

    # ── Distress probability trend chart ─────────────────────────────
    if firm_preds is not None and len(firm_preds) > 1:
        st.markdown("#### Predicted distress probability — 6-month outlook")

        display_preds = firm_preds.copy()
        display_preds["quarter"] = display_preds.apply(
            lambda r: quarter_label(
                int(r["fiscal_year"]), str(r.get("fiscal_period", ""))
            ),
            axis=1,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=display_preds["quarter"],
                y=display_preds["distress_probability"],
                mode="lines+markers",
                line={"color": risk_color(latest_score), "width": 2.5},
                marker={"size": 8},
                name="Distress probability",
                hovertemplate="Quarter: %{x}<br>Probability: %{y:.2%}<extra></extra>",
            )
        )
        fig.add_hline(
            y=0.70, line_dash="dash", line_color=COLORS["high"], opacity=0.4,
            annotation_text="High risk (0.70)", annotation_position="top left",
        )
        fig.add_hline(
            y=0.30, line_dash="dash", line_color=COLORS["medium"], opacity=0.3,
            annotation_text="Medium (0.30)", annotation_position="top left",
        )
        fig.update_yaxes(title_text="Distress probability", range=[-0.05, 1.05], tickformat=".0%")
        fig.update_xaxes(title_text="")
        fig.update_layout(height=280, showlegend=False)
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

    elif firm_preds is not None and len(firm_preds) == 1:
        st.metric(
            "Predicted distress probability",
            f"{latest_score:.2%}",
            help="Single quarter prediction from XGBoost model",
        )

    elif not history.empty and "distress_label" in history.columns:
        st.markdown("#### Distress label — last 8 quarters")
        st.caption("⚠️ Binary labels shown. Probability predictions available for test set (2022–2023).")

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
                x=recent["quarter"], y=recent["distress_label"],
                mode="lines+markers",
                line={"color": risk_color(latest_score), "width": 2},
                marker={"size": 7},
            )
        )
        fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["Healthy", "Distressed"])
        fig.update_layout(height=250, showlegend=False)
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

    # ── Two columns: SHAP drivers + financial snapshot ───────────────
    col_shap, col_snapshot = st.columns(2)

    with col_shap:
        st.markdown("#### Top risk drivers")

        company_shap = get_shap_for_company(shap_df, selected_firm)

        if not company_shap.empty and "top_features_json" in company_shap.columns:
            shap_row = company_shap.iloc[-1]
            features = parse_top_features_json(shap_row["top_features_json"])

            # Extend to top 5 from raw SHAP columns
            shap_cols = [c for c in company_shap.columns if c.startswith("shap_")]
            if shap_cols:
                row_data = company_shap.iloc[-1]
                vals = {c.replace("shap_", ""): float(row_data[c]) for c in shap_cols}
                sorted_feats = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                features = [
                    {"feature": f, "shap_value": v, "rank": i + 1}
                    for i, (f, v) in enumerate(sorted_feats)
                ]

            if features:
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

                if firm_preds is not None and not firm_preds.empty:
                    st.caption(f"Predicted probability: **{latest_score:.2%}**")
            else:
                st.info("No SHAP explanations available for this quarter.")
        else:
            st.info(
                "SHAP data not available for this company. "
                "SHAP values are computed on the test set (2022–2023)."
            )

    with col_snapshot:
        st.markdown("#### Financial snapshot")

        if not history.empty:
            lh = history.iloc[-1]

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
                if col in lh.index:
                    val = lh[col]
                    formatted = formatter(val) if val != 0 else "—"
                    st.markdown(
                        f"""<div style="display:flex;justify-content:space-between;
                        padding:5px 0;border-bottom:0.5px solid rgba(0,0,0,0.07);
                        font-size:13px">
                            <span style="color:#73726c">{label}</span>
                            <span style="font-weight:500">{formatted}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            if firm_preds is not None and not firm_preds.empty:
                ci_low = max(0, latest_score - 0.05)
                ci_high = min(1, latest_score + 0.05)
                st.markdown(
                    f"""<div style="display:flex;justify-content:space-between;
                    padding:5px 0;border-bottom:0.5px solid rgba(0,0,0,0.07);
                    font-size:13px">
                        <span style="color:#73726c">Confidence interval</span>
                        <span style="color:#9c9a92">[{ci_low:.2f}, {ci_high:.2f}]</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown(
                """<div style="display:flex;justify-content:space-between;
                padding:5px 0;font-size:13px">
                    <span style="color:#73726c">Model version</span>
                    <span style="color:#9c9a92">v1</span>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("No financial data available for this company.")