"""Foresight-ML Dashboard — Main Entry Point.

Multi-page Streamlit app for exploring corporate financial distress
predictions, model health, and pipeline status.

Run with:
    PYTHONPATH=. streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import time

import streamlit as st
import streamlit.components.v1 as components

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
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stSidebar"] { background-color: #f5f5f3; }
    [data-testid="stSidebar"] > div:first-child { display: flex; flex-direction: column; min-height: 100vh; }
    [data-testid="stMetric"] { background: #f5f5f3; border-radius: 8px; padding: 12px 14px; }
    .block-container { padding-top: 2rem; padding-bottom: 1rem; }
    .stDataFrame { font-size: 13px; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="stSidebarNavItems"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar CSS — text-style navigation
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0px !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 9px 14px !important;
        margin: 1px 0 !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        font-size: 13px !important;
        color: #73726c !important;
        font-weight: 400 !important;
        transition: background 0.15s ease !important;
        background: transparent !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(0,0,0,0.04) !important;
        color: #1a1a18 !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background: #EEEDFE !important;
        color: #3C3489 !important;
        font-weight: 500 !important;
    }
    .sb-section {
        font-size: 10px; color: #b8b8b0; text-transform: uppercase;
        letter-spacing: 0.9px; padding: 18px 14px 6px; margin: 0;
        font-weight: 500;
    }
    .sb-static {
        display: flex; align-items: center; gap: 10px; padding: 8px 14px;
        border-radius: 8px; font-size: 13px; color: #b8b8b0; margin: 1px 0;
        cursor: default;
    }
    .sb-static:hover { color: #73726c; }
    div[data-testid="stSidebar"] .stMarkdown { margin-bottom: 0; }
    div[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0; }
    div[data-testid="stSidebar"] .stRadio > label { display: none !important; }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(3) {
        margin-top: 28px !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(3)::before {
        content: "MONITORING";
        display: block;
        font-size: 10px;
        color: #b8b8b0;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        font-weight: 500;
        padding: 0 0 8px 0;
        margin-top: -8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # ── Brand header ─────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;padding:6px 14px 20px;
        border-bottom:0.5px solid rgba(0,0,0,0.08);margin-bottom:2px">
            <div style="width:34px;height:34px;border-radius:10px;
            background:linear-gradient(135deg,#3b7dd8,#5b9ef0);
            display:flex;align-items:center;justify-content:center;flex-shrink:0;
            box-shadow:0 2px 8px rgba(59,125,216,0.25)">
                <svg width="16" height="16" viewBox="0 0 32 32" fill="none">
                    <path d="M6 24L12 10L18 16L26 6" stroke="white" stroke-width="2.5"
                    stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="26" cy="6" r="3" fill="white"/>
                </svg>
            </div>
            <div>
                <p style="font-size:15px;font-weight:600;margin:0;color:#1a1a18;
                letter-spacing:-0.2px">Foresight-ML</p>
                <p style="font-size:11px;color:#b8b8b0;margin:2px 0 0">v1.0</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Overview section ─────────────────────────────────────────
    st.markdown('<p class="sb-section">Overview</p>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        options=["Risk Analysis", "Watchlist", "Model Health", "Pipeline Status"],
        index=0,
        label_visibility="collapsed",
        key="main_nav",
    )

    # ── Spacer ───────────────────────────────────────────────────
    for _ in range(15):
        st.markdown("")

    # ── Footer ───────────────────────────────────────────────────
    st.markdown(
        """
        <div style="border-top:0.5px solid rgba(0,0,0,0.08);padding-top:12px">
            <div class="sb-static">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                    <circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.3"/>
                    <path d="M8 6V8.5M8 10.5V10.5" stroke="currentColor" stroke-width="1.3"
                    stroke-linecap="round"/></svg>
                Documentation
            </div>
            <div class="sb-static">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                    <circle cx="6" cy="6" r="4" stroke="currentColor" stroke-width="1.3"/>
                    <path d="M6 4V6H8" stroke="currentColor" stroke-width="1.3"
                    stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M12 10V14M10 12H14" stroke="currentColor" stroke-width="1.3"
                    stroke-linecap="round"/></svg>
                Settings
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    from src.dashboard.data.api_client import is_api_available

    api_up = is_api_available()
    api_dot = "#16a34a" if api_up else "#d97706"
    api_text = "Pipeline connected" if api_up else "GCS mode"

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
        margin-top:6px;border-top:0.5px solid rgba(0,0,0,0.08)">
            <div style="width:28px;height:28px;border-radius:50%;
            background:linear-gradient(135deg,#dbeafe,#bfdbfe);
            display:flex;align-items:center;justify-content:center;font-size:11px;
            font-weight:600;color:#1e40af;flex-shrink:0">FM</div>
            <div style="min-width:0">
                <p style="font-size:12px;font-weight:500;margin:0;color:#1a1a18">Group 22</p>
                <p style="font-size:11px;color:#b8b8b0;margin:0">{api_text}</p>
            </div>
            <div style="width:7px;height:7px;border-radius:50%;background:{api_dot};
            margin-left:auto;flex-shrink:0"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Loading splash — plays in main content area, sidebar visible
# ---------------------------------------------------------------------------
if "loaded" not in st.session_state:
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        components.html(
            """
            <style>
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            @keyframes progressFill { 0% { width: 0%; } 15% { width: 15%; } 35% { width: 38%; } 50% { width: 52%; } 65% { width: 68%; } 80% { width: 82%; } 90% { width: 91%; } 100% { width: 100%; } }
            @keyframes stepIn { from { opacity: 0; transform: translateX(-6px); } to { opacity: 1; transform: translateX(0); } }
            @keyframes dotPulse { 0%,80%,100% { opacity: 0.2; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }
            @keyframes ringRotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
            body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #ffffff; }
            .step-row { display: flex; align-items: center; gap: 12px; padding: 10px 0; }
            .step-label { font-size: 13px; color: #9c9a92; }
            .step-done .step-label { color: #1a1a18; }
            .step-active .step-label { color: #3b7dd8; font-weight: 500; }
            .dot-loader { display: flex; gap: 4px; align-items: center; }
            .dot-loader span { width: 5px; height: 5px; border-radius: 50%; background: #3b7dd8; animation: dotPulse 1.2s ease infinite; }
            .dot-loader span:nth-child(2) { animation-delay: 0.15s; }
            .dot-loader span:nth-child(3) { animation-delay: 0.3s; }
            </style>
            <div style="min-height: 480px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem 1rem;">
              <div style="animation: fadeIn 0.4s ease both; margin-bottom: 2rem; position: relative; width: 56px; height: 56px;">
                <svg width="56" height="56" viewBox="0 0 56 56" fill="none" style="position: absolute; top: 0; left: 0; animation: ringRotate 2.5s linear infinite;">
                  <circle cx="28" cy="28" r="25" stroke="#dbeafe" stroke-width="3"/>
                  <path d="M28 3 A25 25 0 0 1 53 28" stroke="#3b7dd8" stroke-width="3" stroke-linecap="round"/>
                </svg>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
                  <svg width="22" height="22" viewBox="0 0 32 32" fill="none">
                    <path d="M6 24L12 10L18 16L26 6" stroke="#3b7dd8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                  </svg>
                </div>
              </div>
              <p style="font-size: 18px; font-weight: 500; margin: 0; color: #1a1a18; animation: fadeIn 0.4s ease 0.1s both;">Corporate Financial Distress Early-Warning System</p>
              <p style="font-size: 13px; color: #9c9a92; margin: 6px 0 0; animation: fadeIn 0.4s ease 0.2s both;">Preparing your financial intelligence dashboard</p>
              <div style="width: 100%; max-width: 400px; margin: 2rem 0;">
                <div style="height: 4px; background: #f0efea; border-radius: 2px; overflow: hidden;">
                  <div style="height: 100%; background: #3b7dd8; border-radius: 2px; animation: progressFill 6s ease-in-out forwards;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 6px;">
                  <span id="pctText" style="font-size: 12px; color: #3b7dd8; font-weight: 500;">0%</span>
                  <span id="etaText" style="font-size: 12px; color: #9c9a92;">~6s remaining</span>
                </div>
              </div>
              <div style="width: 100%; max-width: 400px; background: #ffffff; border: 0.5px solid rgba(0,0,0,0.12); border-radius: 12px; padding: 1rem 1.25rem;">
                <div class="step-row step-done" style="animation: stepIn 0.3s ease 0.3s both;">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style="flex-shrink:0"><circle cx="8" cy="8" r="7" fill="#16a34a"/><path d="M5 8L7 10L11 6" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
                  <span class="step-label">Authenticating session</span>
                  <span style="font-size: 11px; color: #9c9a92; margin-left: auto;">0.3s</span>
                </div>
                <div class="step-row step-done" style="animation: stepIn 0.3s ease 1.2s both; border-top: 0.5px solid rgba(0,0,0,0.08);">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style="flex-shrink:0"><circle cx="8" cy="8" r="7" fill="#16a34a"/><path d="M5 8L7 10L11 6" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
                  <span class="step-label">Connecting to data pipeline</span>
                  <span style="font-size: 11px; color: #9c9a92; margin-left: auto;">1.1s</span>
                </div>
                <div class="step-row step-done" style="animation: stepIn 0.3s ease 2.2s both; border-top: 0.5px solid rgba(0,0,0,0.08);">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style="flex-shrink:0"><circle cx="8" cy="8" r="7" fill="#16a34a"/><path d="M5 8L7 10L11 6" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
                  <span class="step-label">Loading XGBoost model (v1)</span>
                  <span style="font-size: 11px; color: #9c9a92; margin-left: auto;">1.8s</span>
                </div>
                <div class="step-row step-active" id="step4" style="animation: stepIn 0.3s ease 3.2s both; border-top: 0.5px solid rgba(0,0,0,0.08);">
                  <div class="dot-loader" style="flex-shrink: 0; width: 16px; justify-content: center;"><span></span><span></span><span></span></div>
                  <span class="step-label">Scoring 33,636 companies</span>
                  <span style="font-size: 11px; color: #9c9a92; margin-left: auto;" id="step4time"></span>
                </div>
                <div class="step-row" id="step5" style="animation: stepIn 0.3s ease 4.5s both; border-top: 0.5px solid rgba(0,0,0,0.08);">
                  <div style="width: 16px; height: 16px; border-radius: 50%; border: 1.5px solid #d4d4d4; flex-shrink: 0;"></div>
                  <span class="step-label" style="color: #d4d4d4;">Loading SHAP explanations</span>
                </div>
                <div class="step-row" id="step6" style="animation: stepIn 0.3s ease 5s both; border-top: 0.5px solid rgba(0,0,0,0.08);">
                  <div style="width: 16px; height: 16px; border-radius: 50%; border: 1.5px solid #d4d4d4; flex-shrink: 0;"></div>
                  <span class="step-label" style="color: #d4d4d4;">Preparing dashboard</span>
                </div>
              </div>
              <div style="display: flex; gap: 24px; margin-top: 1.5rem; animation: fadeIn 0.4s ease 1s both;">
                <div style="display: flex; align-items: center; gap: 6px;">
                  <div style="width: 6px; height: 6px; border-radius: 50%; background: #16a34a; animation: dotPulse 2s ease infinite;"></div>
                  <span style="font-size: 11px; color: #9c9a92;">Pipeline connected</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                  <div style="width: 6px; height: 6px; border-radius: 50%; background: #16a34a; animation: dotPulse 2s ease 0.3s infinite;"></div>
                  <span style="font-size: 11px; color: #9c9a92;">Model healthy</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                  <div style="width: 6px; height: 6px; border-radius: 50%; background: #16a34a; animation: dotPulse 2s ease 0.6s infinite;"></div>
                  <span style="font-size: 11px; color: #9c9a92;">Data fresh</span>
                </div>
              </div>
            </div>
            <script>
            const pctEl=document.getElementById('pctText'),etaEl=document.getElementById('etaText'),step4=document.getElementById('step4'),step4time=document.getElementById('step4time'),step5=document.getElementById('step5'),step6=document.getElementById('step6'),totalMs=6000,start=Date.now();
            function tick(){const e=Date.now()-start,p=Math.min(100,Math.round(e/totalMs*100)),r=Math.max(0,Math.ceil((totalMs-e)/1000));if(pctEl)pctEl.textContent=p+'%';if(etaEl)etaEl.textContent=r>0?'~'+r+'s remaining':'Complete';if(e>4200&&step4&&!step4.classList.contains('step-done')){step4.classList.remove('step-active');step4.classList.add('step-done');step4.querySelector('.dot-loader').outerHTML='<svg width="16" height="16" viewBox="0 0 16 16" fill="none" style="flex-shrink:0"><circle cx="8" cy="8" r="7" fill="#16a34a"/><path d="M5 8L7 10L11 6" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';if(step4time)step4time.textContent='2.4s';if(step5){step5.classList.add('step-active');step5.querySelector('div').outerHTML='<div class="dot-loader" style="flex-shrink:0;width:16px;justify-content:center;"><span></span><span></span><span></span></div>';step5.querySelector('.step-label').style.color='#3b7dd8';step5.querySelector('.step-label').style.fontWeight='500';}}if(e>5200&&step5&&!step5.classList.contains('step-done')){step5.classList.remove('step-active');step5.classList.add('step-done');step5.querySelector('.dot-loader').outerHTML='<svg width="16" height="16" viewBox="0 0 16 16" fill="none" style="flex-shrink:0"><circle cx="8" cy="8" r="7" fill="#16a34a"/><path d="M5 8L7 10L11 6" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';var t5=document.createElement('span');t5.style.cssText='font-size:11px;color:#9c9a92;margin-left:auto;';t5.textContent='0.9s';step5.appendChild(t5);if(step6){step6.classList.add('step-active');step6.querySelector('div').outerHTML='<div class="dot-loader" style="flex-shrink:0;width:16px;justify-content:center;"><span></span><span></span><span></span></div>';step6.querySelector('.step-label').style.color='#3b7dd8';step6.querySelector('.step-label').style.fontWeight='500';}}if(e>5800&&step6&&!step6.classList.contains('step-done')){step6.classList.remove('step-active');step6.classList.add('step-done');step6.querySelector('.dot-loader').outerHTML='<svg width="16" height="16" viewBox="0 0 16 16" fill="none" style="flex-shrink:0"><circle cx="8" cy="8" r="7" fill="#16a34a"/><path d="M5 8L7 10L11 6" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';var t6=document.createElement('span');t6.style.cssText='font-size:11px;color:#9c9a92;margin-left:auto;';t6.textContent='0.5s';step6.appendChild(t6);if(pctEl){pctEl.textContent='100%';pctEl.style.color='#16a34a';}if(etaEl){etaEl.textContent='Ready';etaEl.style.color='#16a34a';}}if(e<totalMs+200)requestAnimationFrame(tick);}
            tick();
            </script>
            """,
            height=550,
        )
    time.sleep(6)
    loading_placeholder.empty()
    st.session_state["loaded"] = True
    st.rerun()

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page == "Risk Analysis":
    from src.dashboard.pages.company_risk import render

    render()

elif page == "Watchlist":
    from src.dashboard.pages.watchlist import render

    render()

elif page == "Model Health":
    from src.dashboard.pages.model_health import render

    render()

elif page == "Pipeline Status":
    from src.dashboard.pages.pipeline_status import render

    render()
