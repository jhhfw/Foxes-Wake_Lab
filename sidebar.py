"""Sidebar-Rendering.

Sammelt die Roheingaben der Bedienelemente pro Arbeitsbereich und gibt sie
als Dict zurueck; app.py entscheidet anhand von layout_mode, welchen
Tab-Renderer es mit diesen Werten aufruft. Trennt die reine Eingabesammlung
von der Ergebnisdarstellung (climate_tab.py / simulation_tab.py).
"""

import streamlit as st

from foxes_runner import CUSTOM_TURBINE_LABEL, WAKE_MODEL_OPTIONS


def render_sidebar(turbine_files, hub_height_defaults):
    with st.sidebar:
        layout_mode = st.segmented_control(
            "Arbeitsbereich",
            ["Turbine & Windklima", "Turbinenreihe", "Windpark", "Runner"],
            key="active_workspace",
        )

        if layout_mode == "Turbine & Windklima":
            values = _render_climate_sidebar(turbine_files)
        elif layout_mode == "Runner":
            st.markdown("**Runner**")
            st.info("Die Bedienung erfolgt im Hauptfenster. Konfigurationen werden aus Turbinenreihe oder Windpark uebernommen.")
            values = {}
        else:
            values = _render_simulation_sidebar(layout_mode, turbine_files, hub_height_defaults)

    return layout_mode, values


def _render_climate_sidebar(turbine_files):
    climate_turbine_label = st.selectbox(
        "Turbine",
        list(turbine_files) + [CUSTOM_TURBINE_LABEL],
    )
    custom_name = st.text_input("Name neue Turbine", "StudentTurbine")
    rotor_diameter = st.number_input("Rotordurchmesser [m]", 20.0, 300.0, 126.0, 1.0)
    custom_hub_height = st.number_input("Nabenhoehe [m]", 20.0, 250.0, 100.0, 1.0)
    rated_power_kw = st.number_input("Nennleistung [kW]", 100.0, 30000.0, 5000.0, 100.0)
    cut_in_ws = st.number_input("Einschaltwind [m/s]", 1.0, 8.0, 3.0, 0.5)
    rated_ws = st.number_input("Nennwind [m/s]", 5.0, 18.0, 12.0, 0.5)
    cut_out_ws = st.number_input("Abschaltwind [m/s]", 12.0, 35.0, 25.0, 0.5)
    ct_below_rated = st.slider("ct unter Nennwind", 0.1, 1.0, 0.8, 0.01)
    ct_above_rated = st.slider("ct ab Nennwind", 0.0, 1.0, 0.35, 0.01)

    st.markdown("**Windklima**")
    weibull_a = st.slider("Weibull A [m/s]", 3.0, 15.0, 9.0, 0.1)
    weibull_k = st.slider("Weibull k [-]", 1.0, 4.0, 2.0, 0.1)
    climate_wind_direction = st.slider("Hauptwindrichtung [deg]", 0.0, 360.0, 270.0, 1.0)
    direction_concentration = st.slider("Faktor Hauptwindrichtungs-Konzentration [-]", 0.0, 8.0, 2.0, 0.1)
    n_speed_bins = st.slider("WS-Klassen", 8, 40, 24, 1)
    n_direction_bins = st.slider("WD-Sektoren", 8, 36, 16, 1)
    climate_ti = st.slider("Turbulenzintensitaet [-]", 0.01, 0.30, 0.08, 0.01)
    climate_rho = st.number_input("Luftdichte [kg/m3]", 0.9, 1.4, 1.225, 0.005)
    run_climate_clicked = st.button(
        "FOXES rechnet..." if st.session_state.is_running else "Analysieren",
        type="primary",
        disabled=st.session_state.is_running,
    )

    return {
        "climate_turbine_label": climate_turbine_label,
        "custom_name": custom_name,
        "rotor_diameter": rotor_diameter,
        "custom_hub_height": custom_hub_height,
        "rated_power_kw": rated_power_kw,
        "cut_in_ws": cut_in_ws,
        "rated_ws": rated_ws,
        "cut_out_ws": cut_out_ws,
        "ct_below_rated": ct_below_rated,
        "ct_above_rated": ct_above_rated,
        "weibull_a": weibull_a,
        "weibull_k": weibull_k,
        "climate_wind_direction": climate_wind_direction,
        "direction_concentration": direction_concentration,
        "n_speed_bins": n_speed_bins,
        "n_direction_bins": n_direction_bins,
        "climate_ti": climate_ti,
        "climate_rho": climate_rho,
        "run_climate_clicked": run_climate_clicked,
    }


def _render_simulation_sidebar(layout_mode, turbine_files, hub_height_defaults):
    n_turbines = st.slider("Turbinen", 2, 20, 5, disabled=layout_mode != "Turbinenreihe")
    spacing_x = st.number_input(
        "Abstand x [m]",
        min_value=100.0,
        max_value=3000.0,
        value=800.0,
        step=50.0,
        disabled=layout_mode != "Turbinenreihe",
    )
    spacing_y = st.number_input(
        "Abstand y [m]",
        min_value=-1500.0,
        max_value=1500.0,
        value=0.0,
        step=50.0,
        disabled=layout_mode != "Turbinenreihe",
    )

    wind_speed = st.slider("Windgeschwindigkeit [m/s]", 3.0, 25.0, 9.0, 0.5)
    wind_direction = st.slider("Windrichtung [deg]", 0.0, 360.0, 270.0, 1.0)
    air_density = st.number_input("Luftdichte [kg/m3]", 0.9, 1.4, 1.225, 0.005)

    st.markdown("**Atmosphaere**")
    atmosphere_preset = st.selectbox(
        "Stabilitaet",
        ["neutral", "stabil", "instabil", "benutzerdefiniert"],
    )
    preset_ti = {
        "stabil": 0.04,
        "neutral": 0.08,
        "instabil": 0.12,
        "benutzerdefiniert": 0.08,
    }[atmosphere_preset]
    turbulence_intensity = st.slider(
        "Turbulenzintensitaet [-]",
        0.01,
        0.30,
        preset_ti,
        0.01,
    )
    vertical_profile = st.selectbox(
        "Vertikalprofil",
        ["uniform", "shear", "ABL log"],
    )
    shear_exponent = st.slider(
        "Shear-Exponent [-]",
        0.00,
        0.50,
        0.12 if atmosphere_preset != "stabil" else 0.22,
        0.01,
        disabled=vertical_profile == "uniform",
    )
    roughness_length = st.number_input(
        "Rauigkeitslaenge z0 [m]",
        min_value=0.0001,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format="%.4f",
        disabled=vertical_profile != "ABL log",
    )
    monin_obukhov_length = st.number_input(
        "Monin-Obukhov-Laenge [m]",
        min_value=-1000.0,
        max_value=1000.0,
        value={
            "stabil": 200.0,
            "neutral": 0.0,
            "instabil": -200.0,
            "benutzerdefiniert": 0.0,
        }[atmosphere_preset],
        step=50.0,
        disabled=vertical_profile != "ABL log",
    )
    wake_length_threshold = st.slider(
        "Wake-Laenge: Defizitschwelle [%]",
        0.5,
        10.0,
        2.0,
        0.5,
    )
    st.markdown("**Wake-Laengen anzeigen**")
    show_midline_wake_length = st.checkbox(
        "Mittellinie",
        value=True,
        disabled=layout_mode != "Turbinenreihe",
    )
    show_streamline_wake_length = st.checkbox(
        "Streamline T0",
        value=True,
        disabled=layout_mode != "Turbinenreihe",
    )
    show_contour_wake_length = st.checkbox(
        "Max. Kontur-Laenge",
        value=True,
        disabled=layout_mode != "Turbinenreihe",
    )
    st.markdown("**Partikelschwarm**")
    particle_tracking = st.checkbox(
        "Virtuelle Partikel verfolgen",
        value=False,
        disabled=layout_mode != "Turbinenreihe",
    )
    particle_release_interval_s = st.number_input(
        "Partikelabstand [s]",
        min_value=0.5,
        max_value=20.0,
        value=1.0,
        step=0.5,
        disabled=not particle_tracking,
    )
    particle_duration_s = st.number_input(
        "Simulationsdauer [s]",
        min_value=10.0,
        max_value=600.0,
        value=120.0,
        step=10.0,
        disabled=not particle_tracking,
    )
    particle_dt_s = st.number_input(
        "Zeitschritt [s]",
        min_value=0.25,
        max_value=10.0,
        value=1.0,
        step=0.25,
        disabled=not particle_tracking,
    )
    particle_turbulence_scale = st.slider(
        "Meander-Skalierung [-]",
        0.0,
        3.0,
        1.0,
        0.1,
        disabled=not particle_tracking,
    )
    particle_seed = st.number_input(
        "Zufalls-Seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        disabled=not particle_tracking,
    )

    turbine_label = st.selectbox("Turbine", list(turbine_files))
    wake_label = st.selectbox("Wake-Modell", list(WAKE_MODEL_OPTIONS))
    plane_height = st.slider("Colormap-Hoehe [m]", 20.0, 250.0, hub_height_defaults[turbine_label], 5.0)
    grid_points = st.slider("Colormap-Aufloesung", 25, 150, 50, 5)
    show_wind_arrows = st.checkbox("Windrichtungspfeile", value=True)
    st.markdown("**Heatmap-Skalierung**")
    heatmap_scale_mode = st.radio(
        "Skalierungsmodus",
        ["Automatisch", "Manuell"],
        horizontal=True,
        key="main_heatmap_scale_mode",
    )
    heatmap_ws_max = st.number_input(
        "WS max Farbskala [m/s]",
        min_value=1.0,
        max_value=60.0,
        value=25.0,
        step=0.5,
        disabled=heatmap_scale_mode == "Automatisch",
        key="main_heatmap_ws_max",
    )
    st.markdown("**Dargestellte Flaeche**")
    x_min = st.number_input("x min [m]", value=-500.0, step=100.0)
    x_max = st.number_input("x max [m]", value=max(2500.0, (n_turbines - 1) * spacing_x + 1500.0), step=100.0)
    y_min = st.number_input("y min [m]", value=-1000.0, step=100.0)
    y_max = st.number_input("y max [m]", value=1000.0, step=100.0)
    run_clicked = st.button(
        "FOXES rechnet..." if st.session_state.is_running else "Simulieren",
        type="primary",
        disabled=st.session_state.is_running,
    )
    send_to_runner_clicked = st.button(
        "In Runner uebernehmen",
        use_container_width=True,
        disabled=st.session_state.is_running,
    )

    return {
        "n_turbines": n_turbines,
        "spacing_x": spacing_x,
        "spacing_y": spacing_y,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "air_density": air_density,
        "atmosphere_preset": atmosphere_preset,
        "turbulence_intensity": turbulence_intensity,
        "vertical_profile": vertical_profile,
        "shear_exponent": shear_exponent,
        "roughness_length": roughness_length,
        "monin_obukhov_length": monin_obukhov_length,
        "wake_length_threshold": wake_length_threshold,
        "show_midline_wake_length": show_midline_wake_length,
        "show_streamline_wake_length": show_streamline_wake_length,
        "show_contour_wake_length": show_contour_wake_length,
        "particle_tracking": particle_tracking,
        "particle_release_interval_s": particle_release_interval_s,
        "particle_duration_s": particle_duration_s,
        "particle_dt_s": particle_dt_s,
        "particle_turbulence_scale": particle_turbulence_scale,
        "particle_seed": particle_seed,
        "turbine_label": turbine_label,
        "wake_label": wake_label,
        "plane_height": plane_height,
        "grid_points": grid_points,
        "show_wind_arrows": show_wind_arrows,
        "heatmap_scale_mode": heatmap_scale_mode,
        "heatmap_ws_max": heatmap_ws_max,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "run_clicked": run_clicked,
        "send_to_runner_clicked": send_to_runner_clicked,
    }
