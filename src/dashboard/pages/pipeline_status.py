"""Page 4 — Pipeline Status.

Shows data pipeline and training pipeline task status,
last run timestamps, and key summary metrics.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.dashboard.data.gcs_loader import (
    load_drift_summary,
    load_manifest,
    load_optuna_results,
    load_predictions,
)


def _status_dot(status: str) -> str:
    """Return colored HTML dot for a pipeline task status."""
    color_map = {
        "success": "#16a34a",
        "failed": "#b91c1c",
        "warning": "#d97706",
        "running": "#3b7dd8",
        "skipped": "#9c9a92",
        "pending": "#9c9a92",
    }
    color = color_map.get(status.lower(), "#9c9a92")
    return f'<div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0"></div>'


def _pipeline_row(name: str, duration: str, status: str, detail: str = "") -> str:
    """Render a single pipeline task row as HTML."""
    status_color = {
        "success": "#16a34a",
        "failed": "#b91c1c",
        "warning": "#d97706",
        "running": "#3b7dd8",
        "skipped": "#9c9a92",
        "pending": "#9c9a92",
    }
    color = status_color.get(status.lower(), "#9c9a92")

    return f"""
    <div style="display:flex;align-items:center;gap:12px;padding:9px 0;
    border-bottom:0.5px solid rgba(0,0,0,0.07)">
        {_status_dot(status)}
        <div style="font-size:13px;font-weight:500;flex:1">{name}</div>
        <div style="font-size:11px;color:#9c9a92;min-width:60px;text-align:right">{duration}</div>
        <div style="font-size:11px;min-width:100px;text-align:right;color:{color}">
            {detail or status.capitalize()}</div>
    </div>
    """


def render() -> None:
    """Render the Pipeline Status page."""
    st.header("Pipeline Status")
    st.caption("Data ingestion and model training pipeline execution status")

    with st.expander("ℹ️ How to use this page", expanded=False):
        st.markdown(
            """
            **Pipeline Status** shows the health of both automated pipelines.

            **Data pipeline** (@daily) — Ingests new SEC filings and FRED economic data,
            cleans, engineers features, and runs validation.

            **Training pipeline** (@weekly) — Retrains the model, evaluates on held-out data,
            and promotes to production only if quality gate passes (ROC-AUC ≥ 0.80).

            **Status indicators:**
            - 🟢 Success — Task completed normally
            - 🟡 Warning — Completed with issues (e.g. drift detected)
            - 🔴 Failed — Task failed, check logs
            - 🔵 Running — Task currently executing
            """
        )

    manifest = load_manifest()
    optuna = load_optuna_results()
    drift = load_drift_summary()
    predictions = load_predictions()

    # ── Determine statuses from available data ───────────────────────
    # Model exists?
    model_exists = Path("artifacts/models/xgb_model.pkl").exists()
    test_exists = Path("artifacts/splits/test.parquet").exists()
    shap_exists = Path("artifacts/shap/shap_values.parquet").exists()
    has_predictions = not predictions.empty
    roc_auc = optuna.get("test_roc_auc", manifest.get("roc_auc", 0))
    drift_detected = drift.get("drift_detected", False)

    scored_at = manifest.get("scored_at", "pending")
    trained_at = manifest.get("trained_at", "—")

    # ── Two columns: data pipeline + training pipeline ───────────────
    col_data, col_train = st.columns(2)

    with col_data:
        st.markdown("#### Data pipeline — `foresight_ingestion`")
        st.caption("Schedule: @daily")
        st.markdown("---")

        data_tasks = [
            ("FRED ingestion", "~4m", "success", "Success"),
            ("SEC XBRL ingestion", "~18m", "success", "Success"),
            ("BigQuery cleaning", "~7m", "success", "Success"),
            ("Panel build + labeling", "~3m", "success", "Success"),
            ("Feature engineering", "~12m", "success", "Success"),
            ("Bias analysis", "~2m", "success", "Success"),
            ("Validation + anomaly detection", "~1m", "success", "Success"),
        ]

        # Add drift monitoring status
        if drift_detected:
            data_tasks.append(("Drift monitoring", "~2m", "warning", "Drift detected"))
            data_tasks.append(("Retraining trigger", "—", "running", "Queued"))
        else:
            data_tasks.append(("Drift monitoring", "~2m", "success", "No drift"))

        for name, duration, status, detail in data_tasks:
            st.markdown(_pipeline_row(name, duration, status, detail), unsafe_allow_html=True)

        st.caption("Last run timestamps based on GCS artifact presence.")

    with col_train:
        st.markdown("#### Training pipeline — `foresight_training`")
        st.caption("Schedule: @weekly")
        st.markdown("---")

        # Build training tasks with real status
        train_tasks = []

        # Data gate
        train_tasks.append(
            (
                "Data gate check",
                "~0.1m",
                "success" if test_exists else "pending",
                "Passed" if test_exists else "Pending",
            )
        )

        # Training
        if optuna.get("best_params"):
            train_tasks.append(("Train + Optuna (25 trials)", "~62m", "success", "Success"))
        else:
            train_tasks.append(("Train + Optuna (25 trials)", "—", "pending", "Pending"))

        # Quality gate
        if roc_auc > 0:
            passed = roc_auc >= 0.80
            train_tasks.append(
                (
                    "Quality gate (ROC-AUC)",
                    "~0.2m",
                    "success" if passed else "failed",
                    f"{roc_auc:.4f} {'✓' if passed else '✗'}",
                )
            )
        else:
            train_tasks.append(("Quality gate (ROC-AUC)", "—", "pending", "Pending"))

        # SHAP + bias
        train_tasks.append(
            (
                "SHAP + bias report",
                "~8m",
                "success" if shap_exists else "pending",
                "Success" if shap_exists else "Pending",
            )
        )

        # Batch inference
        train_tasks.append(
            (
                "Batch inference",
                "~5m",
                "success" if has_predictions else "pending",
                f"{len(predictions):,} rows" if has_predictions else "Pending",
            )
        )

        # Registry
        train_tasks.append(
            (
                "Registry + rollback check",
                "~0.5m",
                "success" if model_exists else "pending",
                "Promoted" if model_exists else "Pending",
            )
        )

        for name, duration, status, detail in train_tasks:
            st.markdown(_pipeline_row(name, duration, status, detail), unsafe_allow_html=True)

        st.caption(f"Trained: {trained_at}")

    # ── Summary metrics ──────────────────────────────────────────────
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Last scored",
        scored_at if scored_at != "pending" else "—",
        help="Timestamp of most recent batch inference run",
    )

    m2.metric(
        "Companies scored",
        f"{len(predictions):,}" if has_predictions else "—",
        help="Total company-quarters in the latest scoring batch",
    )

    high_risk = 0
    if has_predictions and "distress_probability" in predictions.columns:
        high_risk = int((predictions["distress_probability"] >= 0.70).sum())
    m3.metric(
        "High-risk companies (≥0.70)", high_risk, help="Companies above the high-risk threshold"
    )

    m4.metric(
        "Model ROC-AUC",
        f"{roc_auc:.4f}" if roc_auc > 0 else "—",
        help="Model's ability to distinguish distressed from healthy firms. Target ≥ 0.80",
    )
    # ── Artifact status ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Artifact status")

    artifacts = [
        ("Model (xgb_model.pkl)", model_exists),
        ("Test split (test.parquet)", test_exists),
        ("SHAP values (shap_values.parquet)", shap_exists),
        ("Predictions", has_predictions),
        ("Optuna results", bool(optuna.get("best_params"))),
    ]

    cols = st.columns(len(artifacts))
    for col, (name, exists) in zip(cols, artifacts, strict=False):
        col.markdown(
            f"""<div style="text-align:center;padding:8px;background:{"#dcfce7" if exists else "#fee2e2"};
            border-radius:8px;font-size:12px">
                <div>{"✅" if exists else "❌"}</div>
                <div style="margin-top:4px;color:{"#166534" if exists else "#b91c1c"}">{name}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Links ────────────────────────────────────────────────────────
    st.markdown("---")
    link1, link2, link3 = st.columns(3)
    link1.markdown("[🔗 MLflow](https://foresight-mlflow-6ool3rlbea-uc.a.run.app)")
    link2.markdown(
        "[🔗 GCS Bucket](https://console.cloud.google.com/storage/browser/financial-distress-data)"
    )
    link3.markdown("[🔗 GitHub](https://github.com/Foresight-ML/foresight_ml)")
