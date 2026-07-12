"""Quasi-statischer FOXES-Runner: UI + Steuerung.

Gleiche Architektur-Aenderung wie im Dynamic Runner (siehe
dynamic_runner_ui.py): der sich wiederholende Teil (naechsten Block
berechnen + Anzeige) laeuft in einem st.fragment(run_every=...) statt ueber
time.sleep()+st.rerun() das gesamte Skript bei jedem Tick neu auszufuehren.
"""

from dataclasses import replace
from contextlib import nullcontext

import numpy as np
import streamlit as st

from foxes_runner import run_single_state
from figures import _runner_power_figure, _runner_wake_loss_figure, _wake_field_figure
from runner_common import _move_angle_towards, _move_towards
from state import _runner_history_dataframe
from dynamic_runner_ui import render_dynamic_runner


def _runner_step(
    target_config,
    block_seconds,
    ramp_substep_seconds,
    max_ramp_substeps,
    ws_rate_per_h,
    wd_rate_per_h,
    ti_rate_per_h,
):
    import foxes

    effective = st.session_state.get("runner_effective_config", target_config)
    n_steps = max(1, int(np.ceil(block_seconds / ramp_substep_seconds)))
    n_steps = min(n_steps, int(max_ramp_substeps))
    substep_seconds = block_seconds / n_steps
    dt_h = substep_seconds / 3600.0

    active_engine = foxes.get_engine(error=False)
    engine_context = nullcontext(active_engine) if active_engine is not None else foxes.Engine.new(engine_type="single")
    with engine_context:
        for _ in range(n_steps):
            next_ws = _move_towards(
                effective.wind_speed,
                target_config.wind_speed,
                ws_rate_per_h * dt_h,
            )
            next_wd = _move_angle_towards(
                effective.wind_direction,
                target_config.wind_direction,
                wd_rate_per_h * dt_h,
            )
            next_ti = _move_towards(
                effective.turbulence_intensity,
                target_config.turbulence_intensity,
                ti_rate_per_h * dt_h,
            )
            effective = replace(
                target_config,
                wind_speed=float(next_ws),
                wind_direction=float(next_wd),
                turbulence_intensity=float(next_ti),
            )

            summary, results_df, layout_df, flow_field = run_single_state(effective)
            energy_mwh = summary["power_mw"] * dt_h
            ambient = summary["ambient_power_mw"]
            wake_loss = 0.0 if ambient <= 0 else (ambient - summary["power_mw"]) / ambient * 100.0

            st.session_state.runner_time_s += substep_seconds
            st.session_state.runner_energy_mwh += energy_mwh
            st.session_state.runner_last_result = (summary, results_df, layout_df, flow_field)

            row = {
                "time_s": st.session_state.runner_time_s,
                "time_h": st.session_state.runner_time_s / 3600.0,
                "power_mw": summary["power_mw"],
                "ambient_power_mw": summary["ambient_power_mw"],
                "wake_loss_percent": wake_loss,
                "efficiency_percent": summary["efficiency_percent"],
                "energy_mwh": energy_mwh,
                "energy_mwh_total": st.session_state.runner_energy_mwh,
                "wind_speed": effective.wind_speed,
                "wind_direction": effective.wind_direction,
                "ti": effective.turbulence_intensity,
                "target_wind_speed": target_config.wind_speed,
                "target_wind_direction": target_config.wind_direction,
                "target_ti": target_config.turbulence_intensity,
            }
            st.session_state.runner_history.append(row)

    st.session_state.runner_effective_config = effective


def render_runner():
    st.subheader("Runner")
    if "runner_base_config" not in st.session_state:
        st.info("Noch keine Konfiguration uebernommen. Wechsle zu Turbinenreihe oder Windpark und nutze dort 'In Runner uebernehmen'.")
        return

    base_config = st.session_state.runner_base_config
    if (
        st.session_state.get("runner_last_result") is not None
        and len(st.session_state.runner_last_result) != 4
    ):
        st.session_state.runner_last_result = None

    st.caption(
        f"Quelle: {st.session_state.get('runner_source', '-')}"
        f" | Turbine: {base_config.turbine_label}"
        f" | Wake-Modell: {base_config.wake_label}"
        f" | Layout: {base_config.layout_mode}"
    )

    runner_mode = st.segmented_control(
        "Runner-Modus",
        ["FOXES quasi-statisch", "Dynamisch experimentell"],
        key="runner_mode",
    )
    if runner_mode == "Dynamisch experimentell":
        render_dynamic_runner(base_config)
        return

    controls = st.columns(5)
    if controls[0].button("Start", type="primary", use_container_width=True):
        st.session_state.runner_active = True
        st.rerun()
    if controls[1].button("Pause", use_container_width=True):
        st.session_state.runner_active = False
    if controls[2].button("Einzelschritt", use_container_width=True):
        st.session_state.runner_do_single_step = True
    if controls[3].button("Stop", use_container_width=True):
        st.session_state.runner_active = False
    if controls[4].button("Reset", use_container_width=True):
        st.session_state.runner_active = False
        st.session_state.runner_time_s = 0.0
        st.session_state.runner_energy_mwh = 0.0
        st.session_state.runner_history = []
        st.session_state.runner_last_result = None
        st.session_state.runner_effective_config = base_config

    live = st.session_state.get("runner_live_config", base_config)
    left, right = st.columns([0.82, 1.18])
    with left:
        st.subheader("Live-Parameter")
        block_minutes = st.number_input("Blockgroesse [min]", 1.0, 180.0, 10.0, 1.0)
        auto_delay_s = st.number_input("Anzeige-Takt [s]", 0.2, 10.0, 1.0, 0.2)
        live_ws = st.slider("Windgeschwindigkeit [m/s]", 3.0, 25.0, float(live.wind_speed), 0.5)
        live_wd = st.slider("Windrichtung [deg]", 0.0, 360.0, float(live.wind_direction), 1.0)
        live_ti = st.slider("Turbulenzintensitaet [-]", 0.01, 0.30, float(live.turbulence_intensity), 0.01)
        live_rho = st.number_input("Luftdichte [kg/m3]", 0.9, 1.4, float(live.air_density), 0.005)
        st.markdown("**Heatmap-Skalierung**")
        runner_heatmap_scale_mode = st.radio(
            "Skalierungsmodus",
            ["Automatisch", "Manuell"],
            horizontal=True,
            key="runner_heatmap_scale_mode",
        )
        runner_heatmap_ws_max = st.number_input(
            "WS max Farbskala [m/s]",
            min_value=1.0,
            max_value=60.0,
            value=25.0,
            step=0.5,
            disabled=runner_heatmap_scale_mode == "Automatisch",
            key="runner_heatmap_ws_max",
        )
        st.markdown("**Rampen fuer sichtbare Aenderungen**")
        ramp_substep_minutes = st.number_input(
            "Interne Zwischenschrittweite [min]",
            0.5,
            30.0,
            2.0,
            0.5,
        )
        max_ramp_substeps = st.slider("Max. FOXES-Zwischenschritte", 1, 12, 5, 1)
        ws_rate_per_h = st.slider("Max. dWS/dt [m/s pro h]", 0.5, 20.0, 4.0, 0.5)
        wd_rate_per_h = st.slider("Max. dWD/dt [deg pro h]", 5.0, 180.0, 45.0, 5.0)
        ti_rate_per_h = st.slider("Max. dTI/dt [1 pro h]", 0.005, 0.200, 0.040, 0.005)

    live_config = replace(
        base_config,
        wind_speed=live_ws,
        wind_direction=live_wd,
        turbulence_intensity=live_ti,
        air_density=live_rho,
        particle_tracking=False,
        grid_points=min(base_config.grid_points, 55),
    )
    st.session_state.runner_live_config = live_config

    def _tick():
        """Sich wiederholender Teil: naechsten Block berechnen (falls aktiv)
        + Statistik/Diagramme neu zeichnen. Laeuft als st.fragment periodisch
        fuer sich alleine, ohne den Rest des Skripts mit auszufuehren.
        """
        do_step = st.session_state.pop("runner_do_single_step", False) or st.session_state.get("runner_active", False)
        if do_step:
            try:
                with st.spinner("Runner rechnet naechsten Block..."):
                    _runner_step(
                        live_config,
                        block_minutes * 60.0,
                        ramp_substep_minutes * 60.0,
                        max_ramp_substeps,
                        ws_rate_per_h,
                        wd_rate_per_h,
                        ti_rate_per_h,
                    )
                    st.session_state.runner_error = None
            except Exception as exc:
                st.session_state.runner_active = False
                st.session_state.runner_error = str(exc)

        effective = st.session_state.get("runner_effective_config", live_config)
        if st.session_state.runner_last_result is None and not st.session_state.get("runner_error"):
            try:
                with st.spinner("Runner berechnet initiales Windfeld..."):
                    summary, results_df, layout_df, flow_field = run_single_state(effective)
                    st.session_state.runner_last_result = (summary, results_df, layout_df, flow_field)
            except Exception as exc:
                st.session_state.runner_error = str(exc)

        if st.session_state.get("runner_error"):
            st.error(st.session_state.runner_error)

        history_df = _runner_history_dataframe()
        total_time_h = st.session_state.get("runner_time_s", 0.0) / 3600.0
        total_energy_mwh = st.session_state.get("runner_energy_mwh", 0.0)
        last_power = 0.0 if history_df.empty else float(history_df["power_mw"].iloc[-1])
        mean_power = 0.0 if history_df.empty else float(history_df["power_mw"].mean())
        last_day = history_df[history_df["time_h"] >= max(total_time_h - 24.0, 0.0)]
        last_day_energy = 0.0 if last_day.empty else float(last_day["energy_mwh"].sum())

        with right:
            st.subheader("Statistik")
            mcols = st.columns(4)
            mcols[0].metric("Simulationszeit", f"{total_time_h:.2f} h")
            mcols[1].metric("Aktuelle Leistung", f"{last_power:.2f} MW")
            mcols[2].metric("Mittlere Leistung", f"{mean_power:.2f} MW")
            mcols[3].metric("Ertrag seit Start", f"{total_energy_mwh:.2f} MWh")
            mcols = st.columns(3)
            mcols[0].metric("Ertrag letzte 24 h", f"{last_day_energy:.2f} MWh")
            if not history_df.empty:
                mcols[1].metric("Mittlerer Wake-Verlust", f"{history_df['wake_loss_percent'].mean():.1f} %")
                mcols[2].metric("Max. Leistung", f"{history_df['power_mw'].max():.2f} MW")
            else:
                mcols[1].metric("Mittlerer Wake-Verlust", "0.0 %")
                mcols[2].metric("Max. Leistung", "0.00 MW")
            mcols = st.columns(3)
            mcols[0].metric("Wirksame WS", f"{effective.wind_speed:.2f} m/s", f"Ziel {live_ws:.2f}")
            mcols[1].metric("Wirksame WD", f"{effective.wind_direction:.0f} deg", f"Ziel {live_wd:.0f}")
            mcols[2].metric("Wirksame TI", f"{effective.turbulence_intensity:.3f}", f"Ziel {live_ti:.3f}")

        plot_left, plot_right = st.columns([1.25, 0.95])
        with plot_left:
            st.subheader("Leistungsverlauf")
            st.plotly_chart(_runner_power_figure(history_df), use_container_width=True)
        with plot_right:
            st.subheader("Wake-Verlust")
            st.plotly_chart(_runner_wake_loss_figure(history_df), use_container_width=True)

        if st.session_state.runner_last_result is not None:
            _summary, results_df, layout_df, flow_field = st.session_state.runner_last_result
            st.subheader("Aktuelles Windfeld")
            st.plotly_chart(
                _wake_field_figure(
                    flow_field,
                    layout_df,
                    show_wind_arrows=True,
                    show_midline_wake_length=base_config.layout_mode == "Turbinenreihe",
                    show_streamline_wake_length=base_config.layout_mode == "Turbinenreihe",
                    show_contour_wake_length=base_config.layout_mode == "Turbinenreihe",
                    show_particles=False,
                    heatmap_scale_mode=runner_heatmap_scale_mode,
                    heatmap_ws_max=runner_heatmap_ws_max,
                    height=620,
                ),
                use_container_width=True,
            )
            st.subheader("Aktuelle Turbinenwerte")
            st.dataframe(results_df, use_container_width=True, hide_index=True, height=240)

    run_every = auto_delay_s if st.session_state.get("runner_active", False) else None
    st.fragment(run_every=run_every)(_tick)()
