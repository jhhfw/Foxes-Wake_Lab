"""Zentrale Stelle fuer das App-Styling.

Vorher stand dieser CSS-Block als ~90-zeiliger st.markdown(..., unsafe_allow_html=True)
Aufruf mitten in app.py. Reine Kosmetik, hat aber mit der eigentlichen App-Logik nichts
zu tun -> eigenes Modul, damit app.py uebersichtlich bleibt.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
div[data-testid="stDataFrame"] [role="columnheader"],
div[data-testid="stDataFrame"] [role="gridcell"] {
    justify-content: center;
    text-align: center;
}
div[data-testid="stDataFrame"] canvas {
    margin-left: auto;
    margin-right: auto;
}
div[data-testid="stDataFrame"] [data-testid="stDataFrameResizableColumnHeader"],
div[data-testid="stDataFrame"] [data-testid="stDataFrameColumnHeader"] {
    justify-content: center;
    text-align: center;
}
div[data-testid="stDataFrame"] [data-testid="stDataFrameResizableColumnHeader"] p,
div[data-testid="stDataFrame"] [data-testid="stDataFrameColumnHeader"] p {
    width: 100%;
    text-align: center;
}
div[data-testid="stDataFrame"] {
    border: 2px solid #2f2f2f;
    border-radius: 4px;
    overflow: hidden;
}
div[data-testid="stDataFrame"] [role="columnheader"],
div[data-testid="stDataFrame"] [role="gridcell"] {
    border-right: 1px solid rgba(47, 47, 47, 0.28);
    border-bottom: 1px solid rgba(47, 47, 47, 0.22);
}
div[data-testid="stDataFrame"] [role="columnheader"] {
    border-bottom: 2px solid #2f2f2f;
}
section[data-testid="stSidebar"] {
    font-size: 1.04rem;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown strong {
    font-size: 1.02rem;
    font-weight: 700;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 1.03rem;
    font-weight: 700;
    color: #1f2933;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] {
    font-size: 1.02rem;
    font-weight: 600;
}
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {
    min-width: 2.15rem;
    min-height: 2.15rem;
    border: 1px solid #5f6b76;
    border-radius: 6px;
    background: linear-gradient(#ffffff, #dfe6ec);
    box-shadow: 0 2px 0 #9aa7b2, 0 1px 3px rgba(0, 0, 0, 0.18);
    color: #111827;
    transition: transform 80ms ease, box-shadow 80ms ease, background 80ms ease;
}
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button:hover {
    background: linear-gradient(#ffffff, #d3dde6);
    border-color: #394956;
}
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button:active {
    transform: translateY(2px);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.22);
    background: linear-gradient(#c8d3dc, #eef3f7);
}
section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    font-size: 1.03rem;
    font-weight: 700;
    border-radius: 7px;
    box-shadow: 0 2px 0 rgba(31, 41, 51, 0.35);
    transition: transform 80ms ease, box-shadow 80ms ease;
}
section[data-testid="stSidebar"] button[kind="primary"]:active,
section[data-testid="stSidebar"] div[data-testid="stButton"] button:active {
    transform: translateY(2px);
    box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.24);
}
</style>
"""


def inject_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
