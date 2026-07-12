"""Hauptbereich fuer die Arbeitsbereiche 'Turbinenreihe' und 'Windpark'."""

import streamlit as st

from foxes_runner import SimulationConfig, run_single_state
from figures import _wake_field_figure
from state import ensure_result, _prepare_runner_from_config


def render_simulation_tab(layout_mode, values, hub_height_defaults):
    default_hub_height = hub_height_defaults[values["turbine_label"]]
    hub_heights = (
        tuple([default_hub_height] * values["n_turbines"]) if layout_mode == "Turbinenreihe" else ()
    )

    config = SimulationConfig(
        layout_mode="Turbinenreihe" if layout_mode == "Turbinenreihe" else "Windpark",
        n_turbines=values["n_turbines"],
        spacing_x=values["spacing_x"],
        spacing_y=values["spacing_y"],
        hub_heights=hub_heights,
        plane_height=values["plane_height"],
        grid_points=values["grid_points"],
        x_min=values["x_min"],
        x_max=values["x_max"],
        y_min=values["y_min"],
        y_max=values["y_max"],
        wind_speed=values["wind_speed"],
        wind_direction=values["wind_direction"],
        turbulence_intensity=values["turbulence_intensity"],
        air_density=values["air_density"],
        atmosphere_preset=values["atmosphere_preset"],
        vertical_profile=values["vertical_profile"],
        shear_exponent=values["shear_exponent"],
        roughness_length=values["roughness_length"],
        monin_obukhov_length=values["monin_obukhov_length"],
        wake_length_threshold=values["wake_length_threshold"],
        particle_tracking=values["particle_tracking"],
        particle_release_interval_s=values["particle_release_interval_s"],
        particle_duration_s=values["particle_duration_s"],
        particle_dt_s=values["particle_dt_s"],
        particle_turbulence_scale=values["particle_turbulence_scale"],
        particle_seed=values["particle_seed"],
        turbine_label=values["turbine_label"],
        wake_label=values["wake_label"],
        # Nur berechnen, was auch angezeigt wird (siehe calculate_wake_length_metrics
        # in foxes_runner.py) - im Windpark-Modus werden diese Metriken nie
        # angezeigt und dort automatisch uebersprungen, unabhaengig von den
        # Checkbox-Werten hier.
        compute_streamline_wake_length=values["show_streamline_wake_length"],
        compute_contour_wake_length=values["show_contour_wake_length"],
    )

    if values["x_min"] >= values["x_max"] or values["y_min"] >= values["y_max"]:
        st.error("Bitte waehle x min < x max und y min < y max.")
        st.stop()

    if values["send_to_runner_clicked"]:
        _prepare_runner_from_config(config, layout_mode)
        st.rerun()

    summary, results_df, layout_df, flow_field = ensure_result(
        config=config,
        config_key="last_config",
        result_key="sim_result",
        error_key="sim_error",
        compute_fn=run_single_state,
        run_clicked=values["run_clicked"],
        spinner_text="FOXES rechnet...",
        waiting_text="Starte eine Simulation.",
    )

    if "wd" not in flow_field:
        st.session_state.is_running = True
        st.rerun()

    st.subheader("Turbinenwerte")
    st.dataframe(results_df, use_container_width=True, hide_index=True, height=260)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Parkleistung", f"{summary['power_mw']:.2f} MW")
    metric_cols[1].metric("Ohne Wake-Verluste", f"{summary['ambient_power_mw']:.2f} MW")
    metric_cols[2].metric("Wirkungsgrad", f"{summary['efficiency_percent']:.1f} %")

    if st.session_state.last_config.layout_mode == "Turbinenreihe":
        metric_cols = st.columns(3)
        metric_cols[0].metric(
            f"Wake-Laenge > {st.session_state.last_config.wake_length_threshold:.1f} %",
            f"{summary['wake_length_m']:.0f} m",
        )
        metric_cols[1].metric("Max. Defizit", f"{summary['wake_max_deficit_percent']:.1f} %")
        metric_cols[2].metric("Min. WS in Mittellinie", f"{summary['wake_min_ws']:.2f} m/s")
        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Wake-Laenge Streamline T0",
            f"{summary['wake_streamline_length_m']:.0f} m",
        )
        metric_cols[1].metric(
            "Max. Kontur-Laenge",
            f"{summary['wake_contour_max_length_m']:.0f} m",
        )
        metric_cols[2].metric(
            "Konturflaeche",
            f"{summary['wake_contour_area_m2'] / 1_000_000:.2f} km2",
        )
        if st.session_state.last_config.particle_tracking:
            metric_cols = st.columns(3)
            metric_cols[0].metric("Partikelbahnen", f"{summary.get('particle_path_count', 0)}")
            metric_cols[1].metric("Laterale Streuung", f"{summary.get('particle_lateral_spread_m', 0.0):.1f} m")
            metric_cols[2].metric("Vertikale Streuung", f"{summary.get('particle_vertical_spread_m', 0.0):.1f} m")

    st.subheader("Windgeschwindigkeit")
    st.plotly_chart(
        _wake_field_figure(
            flow_field,
            layout_df,
            show_wind_arrows=values["show_wind_arrows"],
            show_midline_wake_length=values["show_midline_wake_length"],
            show_streamline_wake_length=values["show_streamline_wake_length"],
            show_contour_wake_length=values["show_contour_wake_length"],
            show_particles=True,
            heatmap_scale_mode=values["heatmap_scale_mode"],
            heatmap_ws_max=values["heatmap_ws_max"],
            height=680,
        ),
        use_container_width=True,
    )
