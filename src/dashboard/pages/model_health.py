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

    # ── Normalize drift fields (handles both old and new schemas) ─────
    drift_detected = drift.get("dataset_drift", drift.get("drift_detected", False))
    drift_date = drift.get("date", drift.get("report_date", "No report yet"))
    n_drifted = drift.get("n_drifted_features", len(drift.get("drifted_features", [])))
    drifted_features = drift.get("drifted_features", [])
    features_checked = drift.get("n_features_analyzed", drift.get("total_features_checked", "—"))
    drift_share = drift.get("drift_share", 0)
    retrain_triggered = drift.get("retrain_triggered", False)

    # ── Drift alert banner ───────────────────────────────────────────
    if drift_detected:
        st.warning(
            f"Dataset drift detected on {drift_date} — "
            f"{n_drifted} feature(s) drifted out of {features_checked} analyzed. "
            f"{'Retraining triggered.' if retrain_triggered else 'Monitoring for retraining threshold.'}",
            icon="⚠️",
        )
    else:
        st.success("No dataset drift detected. Model is current.", icon="✅")

    # ── Two columns: model card + drift status ───────────────────────
    col_model, col_drift = st.columns(2)

    with col_model:
        st.markdown("#### Production model")
        st.markdown("---")

        roc_auc = manifest.get("roc_auc", optuna.get("test_roc_auc", 0))
        model_fields = [
            ("Model name", manifest.get("model_name", "foresight")),
            ("Version", manifest.get("model_version", "v1")),
            ("ROC-AUC", f"{roc_auc:.4f}"),
            ("Trained", manifest.get("trained_at", "—")),
            ("Last scored", manifest.get("scored_at", "pending")),
            ("MLflow run", manifest.get("mlflow_run_id", "—")),
            ("Status", "🟢 Production" if roc_auc >= 0.80 else "🔴 Degraded"),
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

        # Optuna hyperparameters
        best_params = optuna.get("best_params", {})
        if best_params:
            st.markdown("##### Hyperparameters")
            for param, value in best_params.items():
                st.markdown(
                    f"""<div style="display:flex;justify-content:space-between;padding:4px 0;
                    border-bottom:0.5px solid rgba(0,0,0,0.05);font-size:12px">
                        <span style="color:#9c9a92">{param}</span>
                        <span>{value}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Optuna baseline comparison
        baseline_roc = optuna.get("baseline_val_roc", 0)
        test_roc = optuna.get("test_roc_auc", 0)
        if baseline_roc > 0 and test_roc > 0:
            improvement = test_roc - baseline_roc
            st.markdown(
                f"""<div style="display:flex;justify-content:space-between;padding:6px 0;
                border-top:0.5px solid rgba(0,0,0,0.07);font-size:12px;margin-top:8px">
                    <span style="color:#9c9a92">Baseline → Tuned</span>
                    <span style="color:{'#16a34a' if improvement > 0 else '#b91c1c'}">
                    {baseline_roc:.4f} → {test_roc:.4f} ({'+' if improvement > 0 else ''}{improvement:.4f})</span>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("[Open MLflow ↗](https://foresight-mlflow-6ool3rlbea-uc.a.run.app)")

    with col_drift:
        st.markdown("#### Drift monitor")
        st.markdown("---")

        drift_bg = "#fee2e2" if drift_detected else "#dcfce7"
        drift_color = "#b91c1c" if drift_detected else "#166534"
        drift_label = "Detected" if drift_detected else "None"

        drift_rows = [
            ("Dataset drift",
             f'<span style="background:{drift_bg};color:{drift_color};'
             f'padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500">'
             f'{drift_label}</span>'),
            ("Report date", str(drift_date)),
            ("Features analyzed", str(features_checked)),
            ("Features drifted", str(n_drifted)),
            ("Drift share", f"{drift_share:.0%}" if isinstance(drift_share, int | float) else str(drift_share)),
            ("Retrain triggered", "Yes" if retrain_triggered else "No"),
        ]

        for label, value in drift_rows:
            st.markdown(
                f"""<div style="display:flex;justify-content:space-between;padding:6px 0;
                border-bottom:0.5px solid rgba(0,0,0,0.07);font-size:13px">
                    <span style="color:#73726c">{label}</span>
                    <span>{value}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        # Drifted features list
        if drifted_features:
            st.markdown("##### Top drifted features (PSI)")
            for feat in drifted_features[:10]:
                if isinstance(feat, dict):
                    name = feat.get("feature", str(feat))
                    psi = feat.get("psi", 0)
                    color = (
                        COLORS["high"] if psi > 0.25
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

    # ── Prediction distribution ──────────────────────────────────────
    if not predictions.empty and "distress_probability" in predictions.columns:
        st.markdown("---")
        st.markdown("#### Prediction distribution")

        probs = predictions["distress_probability"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Total scored", f"{len(predictions):,}",
            help="Number of company-quarters scored by the model",
        )
        m2.metric(
            "Mean probability", f"{probs.mean():.2%}",
            help="Average predicted distress probability across all companies",
        )
        m3.metric(
            "High risk (≥0.70)", f"{(probs >= 0.70).sum():,}",
            help="Companies with >70% predicted chance of distress",
        )
        m4.metric(
            "Median probability", f"{probs.median():.4f}",
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
            x=0.70, line_dash="dash", line_color=COLORS["high"], opacity=0.6,
            annotation_text="High risk", annotation_position="top right",
        )
        fig.add_vline(
            x=0.30, line_dash="dash", line_color=COLORS["medium"], opacity=0.4,
            annotation_text="Medium", annotation_position="top right",
        )
        fig.update_xaxes(title_text="Distress probability", range=[0, 1])
        fig.update_yaxes(title_text="Number of companies")
        fig.update_layout(height=280, showlegend=False)
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Most companies cluster near 0 (healthy). The tail toward 1.0 represents "
            "firms identified as high distress risk."
        )

    # ── Slice performance table ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Slice performance")

    if not slice_perf.empty:
        display_cols = [
            c for c in [
                "dimension", "slice", "sample_count", "roc_auc",
                "recall_at_5pct", "precision_at_5pct", "brier_score",
            ]
            if c in slice_perf.columns
        ]
        st.dataframe(
            slice_perf[display_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Slice performance data not yet available. Run the evaluation pipeline to generate per-slice metrics.")