"""FOXES Wake Lab - Streamlit-Einstiegspunkt.

Nur noch Seiten-Setup, Sidebar-Dispatch und Weiterleitung an den passenden
Tab-Renderer. Die eigentliche Logik steckt in:
- foxes_runner.py       FOXES-Simulation (mit Caching, siehe dortige Kommentare)
- dynamic_runner.py     Experimentelles Wake-Paket-Modell (vektorisiert)
- figures.py            Plotly-Figures
- state.py              Session-State-Helfer
- runner_common.py      Geteilte Rampen-Mathematik
- sidebar.py            Sidebar-Widgets -> Werte-Dict
- climate_tab.py        Tab 'Turbine & Windklima'
- simulation_tab.py     Tabs 'Turbinenreihe' / 'Windpark'
- runner_ui.py          Tab 'Runner' (quasi-statisch, st.fragment-Loop)
- dynamic_runner_ui.py  Tab 'Runner' -> 'Dynamisch experimentell' (st.fragment-Loop)
- styles.py             CSS
"""

import streamlit as st

import styles
from foxes_runner import available_turbines, default_hub_heights
from sidebar import render_sidebar
from state import init_app_state
from climate_tab import render_climate_tab
from simulation_tab import render_simulation_tab
from runner_ui import render_runner

st.set_page_config(page_title="FOXES Wake Lab", layout="wide")
st.title("FOXES Wake Lab")
styles.inject_custom_css()

init_app_state()

turbine_files = available_turbines()
hub_height_defaults = default_hub_heights(turbine_files)

layout_mode, values = render_sidebar(turbine_files, hub_height_defaults)

if layout_mode == "Runner":
    render_runner()
    st.stop()
elif layout_mode == "Turbine & Windklima":
    render_climate_tab(values)
else:
    render_simulation_tab(layout_mode, values, hub_height_defaults)
