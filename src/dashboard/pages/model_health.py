"""Page 3 — Model Health.

Current production model card, drift status, prediction distribution
histogram, and per-slice performance table.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data.gcs_loader import (
    load_drift_summary,
    load_manifest,
    load_optuna_results,
    load_predictions,
    load_slice_performance,
)
from src.dashboard.utils import COLORS, apply_chart_theme


def render() -> None:
    """Render the Model Health page."""
    st.header("Model Health")
    st.caption("Production model status, drift monitoring, and prediction quality")

    with st.expander("ℹ️ How to use this page", expanded=False):
        st.markdown(
            """
            **Model Health** monitors the production model's quality and data freshness.

            **Key metrics:**
            - **ROC-AUC** — Model's ability to rank distressed vs healthy firms (target ≥ 0.80)
            - **Drift status** — Whether input data distribution has shifted since training
            - **Prediction distribution** — How scores are spread across all companies

            **Alerts:**
            - ⚠️ Drift detected → model may need retraining
            - 🔴 ROC-AUC below 0.80 → model quality degraded
            """
        )

    manifest = load_manifest()
    optuna = load_optuna_results()
    drift = load_drift_summary()
    predictions = load_predictions()
    slice_perf = load_slice_performance()

    # ── Drift alert banner ───────────────────────────────────────────
    if drift.get("drift_detected"):
        n_drifted = len(drift.get("drifted_features", []))
        report_date = drift.get("report_date", "unknown date")
        st.warning(
            f"⚠️ Dataset drift detected on {report_date} — "
            f"{n_drifted} feature(s) above PSI threshold. "
            f"Retraining may be triggered automatically.",
            icon="⚠️",
        )
    else:
        st.success("No dataset drift detected. Model is current.", icon="✅")

    # ── Two columns: model card + drift status ───────────────────────
    col_model, col_drift = st.columns(2)

    with col_model:
        st.markdown("#### Current production model")
        st.markdown("---")

        model_fields = [
            ("Model name", manifest.get("model_name", "foresight-xgboost")),
            ("Version", manifest.get("model_version", "v1")),
            ("Test ROC-AUC", f"{optuna.get('test_roc_auc', manifest.get('roc_auc', 0)):.4f}"),
            ("Baseline val ROC-AUC", f"{optuna.get('baseline_val_roc', 0):.4f}"),
            ("Trained", manifest.get("trained_at", "—")),
            ("Scored", manifest.get("scored_at", "pending")),
            ("MLflow run", manifest.get("mlflow_run_id", "—")),
            ("Status", "🟢 Production"),
        ]

        for label, value in model_fields:
            st.markdown(
                f"""<div style="display:flex;justify-content:space-between;padding:6px 0;
                border-bottom:0.5px solid rgba(0,0,0,0.07);font-size:13px">
                    <span style="color:#73726c">{label}</span>
                    <span style="font-weight:500">{value}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        best_params = optuna.get("best_params", {})
        if best_params:
            st.markdown("##### Best hyperparameters")
            for param, value in best_params.items():
                st.markdown(
                    f"""<div style="display:flex;justify-content:space-between;padding:4px 0;
                    border-bottom:0.5px solid rgba(0,0,0,0.05);font-size:12px">
                        <span style="color:#9c9a92">{param}</span>
                        <span>{value}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("[Open MLflow ↗](https://foresight-mlflow-6ool3rlbea-uc.a.run.app)")

    with col_drift:
        st.markdown("#### Drift monitor")
        st.markdown("---")

        drift_detected = drift.get("drift_detected", False)
        report_date = drift.get("report_date", "No report yet")

        st.markdown(
            f"""<div style="display:flex;justify-content:space-between;padding:6px 0;
            border-bottom:0.5px solid rgba(0,0,0,0.07);font-size:13px">
                <span style="color:#73726c">Dataset drift</span>
                <span style="background:{'#fee2e2' if drift_detected else '#dcfce7'};
                color:{'#b91c1c' if drift_detected else '#166534'};
                padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500">
                {'Detected' if drift_detected else 'None'}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div style="display:flex;justify-content:space-between;padding:6px 0;
            border-bottom:0.5px solid rgba(0,0,0,0.07);font-size:13px">
                <span style="color:#73726c">Report date</span>
                <span>{report_date}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        drifted = drift.get("drifted_features", [])
        if drifted:
            st.markdown("##### Top drifted features (PSI)")
            for feat in drifted[:10]:
                if isinstance(feat, dict):
                    name = feat.get("feature", str(feat))
                    psi = feat.get("psi", 0)
                    color = (
                        COLORS["high"]
                        if psi > 0.25
                        else (COLORS["medium"] if psi > 0.10 else COLORS["low"])
                    )
                    st.markdown(
                        f"""<div style="display:flex;justify-content:space-between;padding:4px 0;
                        border-bottom:0.5px solid rgba(0,0,0,0.05);font-size:12px">
                            <span>{name}</span>
                            <span style="font-weight:500;color:{color}">{psi:.3f}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"- {feat}")
        else:
            st.info("No drifted features to display.")

        st.markdown(
            f"""<div style="display:flex;justify-content:space-between;padding:6px 0;font-size:13px">
                <span style="color:#73726c">Features checked</span>
                <span>{drift.get('total_features_checked', '—')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Prediction distribution ──────────────────────────────────────
    if not predictions.empty and "distress_probability" in predictions.columns:
        st.markdown("---")
        st.markdown("#### Prediction distribution")

        probs = predictions["distress_probability"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Total scored",
            f"{len(predictions):,}",
            help="Number of company-quarters scored by the model",
        )
        m2.metric(
            "Mean probability",
            f"{probs.mean():.2%}",
            help="Average predicted distress probability across all companies",
        )
        m3.metric(
            "High risk (≥0.70)",
            f"{(probs >= 0.70).sum():,}",
            help="Companies with >70% predicted chance of distress in 6 months",
        )
        m4.metric(
            "Median probability",
            f"{probs.median():.4f}",
            help="Middle value — 50% of companies are above, 50% below",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=probs,
                nbinsx=50,
                marker={"color": "#3b7dd8", "line": {"color": "white", "width": 0.5}},
                hovertemplate="Probability: %{x:.2f}<br>Count: %{y}<extra></extra>",
            )
        )
        fig.add_vline(
            x=0.70,
            line_dash="dash",
            line_color=COLORS["high"],
            opacity=0.6,
            annotation_text="High risk",
            annotation_position="top right",
        )
        fig.add_vline(
            x=0.30,
            line_dash="dash",
            line_color=COLORS["medium"],
            opacity=0.4,
            annotation_text="Medium",
            annotation_position="top right",
        )
        fig.update_xaxes(title_text="Distress probability", range=[0, 1])
        fig.update_yaxes(title_text="Number of companies")
        fig.update_layout(height=280, showlegend=False)
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Most companies cluster near 0 (healthy). The tail toward 1.0 represents "
            "firms the model identifies as high distress risk."
        )

    # ── Slice performance table ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Slice performance — test set (2022–2023)")

    if not slice_perf.empty:
        display_cols = [
            c
            for c in [
                "dimension",
                "slice",
                "sample_count",
                "roc_auc",
                "recall_at_5pct",
                "precision_at_5pct",
                "brier_score",
            ]
            if c in slice_perf.columns
        ]

        st.dataframe(
            slice_perf[display_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Slice performance data not available. "
            "Run evaluation pipeline to generate per-slice metrics."
        )
        st.caption(
            "Expected slices: company size (small/mid/large/mega), "
            "sector proxy, time period (pre/post 2016), macro regime (high/low fed funds)."
        )
