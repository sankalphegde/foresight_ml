"""Foresight-ML Dashboard — Main Entry Point.

Multi-page Streamlit app for exploring corporate financial distress
predictions, model health, and pipeline status.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Foresight-ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — matches the HTML mockup aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean font */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Sidebar branding */
    [data-testid="stSidebar"] {
        background-color: #f5f5f3;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f5f5f3;
        border-radius: 8px;
        padding: 12px 14px;
    }

    /* Tighter spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Table styling */
    .stDataFrame {
        font-size: 13px;
    }

    /* Hide default hamburger menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Foresight-ML")
    st.caption("Predicting corporate distress 6 months ahead")
    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "🔍 Company Risk",
            "🔴 Watchlist",
            "🩺 Model Health",
            "⚙️ Pipeline Status",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # API status indicator
    from src.dashboard.data.api_client import is_api_available

    api_up = is_api_available()
    if api_up:
        st.success("API: Connected", icon="✅")
    else:
        st.warning("API: Offline — using GCS data", icon="⚠️")

    st.caption("v1.0 · prod")

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page == "🔍 Company Risk":
    from src.dashboard.pages.company_risk import render

    render()

elif page == "🔴 Watchlist":
    from src.dashboard.pages.watchlist import render

    render()

elif page == "🩺 Model Health":
    from src.dashboard.pages.model_health import render

    render()

elif page == "⚙️ Pipeline Status":
    from src.dashboard.pages.pipeline_status import render

    render()
