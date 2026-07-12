"""Experimenteller Dynamic Runner: UI + Steuerung.

Architektur-Aenderung gegenueber der urspruenglichen Version: Der
Animations-Loop lief vorher ueber `time.sleep(...)` + `st.rerun()` am Ende
der Renderfunktion. Das fuehrt bei JEDEM Tick das GESAMTE app.py-Skript neu
aus - kompletter Sidebar-Aufbau, alle Widgets, alles - nur um am Ende ein
paar Diagramme neu zu zeichnen. Die sich wiederholende Berechnung
("naechster Zeitschritt" + Anzeige) laeuft jetzt stattdessen in einem
`st.fragment(run_every=...)`: Streamlit fuehrt dabei periodisch NUR diesen
Teil erneut aus, nicht das gesamte Skript. Die Regler/Buttons links bleiben
normale Widgets ausserhalb des Fragments.
"""

from dataclasses import replace

import numpy as np
import streamlit as st

from dynamic_runner import (
    DynamicRunnerParameters,
    add_gust_front,
    create_dynamic_runner_state,
    dynamic_history_dataframe,
    evaluate_dynamic_state,
    step_dynamic_runner,
)
from foxes_runner import run_single_state
from figures import (
    _runner_power_figure,
    _runner_wake_loss_figure,
    _wake_field_comparison_figure,
    _wake_field_figure,
)
from runner_common import _move_angle_towards, _move_towards


# ---------------------------------------------------------------------------
# Kalibrierungs-/Vergleichs-Helfer (nur vom Dynamic Runner verwendet)
# ---------------------------------------------------------------------------

def _flow_deficit_percent(flow_field):
    ws = np.asarray(flow_field["ws"], dtype=float)
    amb_ws = np.asarray(flow_field.get("amb_ws", np.nan), dtype=float)
    if amb_ws.shape != ws.shape or np.all(~np.isfinite(amb_ws)):
        amb_ws = np.full_like(ws, float(np.nanmax(ws)))
    deficit = np.zeros_like(ws, dtype=float)
    valid = np.isfinite(ws) & np.isfinite(amb_ws) & (amb_ws > 1e-9)
    np.divide(amb_ws - ws, amb_ws, out=deficit, where=valid)
    return np.maximum(0.0, deficit * 100.0)


def _interpolate_grid_to_target(source_x, source_y, source_values, target_x, target_y):
    source_x = np.asarray(source_x, dtype=float)
    source_y = np.asarray(source_y, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    target_y = np.asarray(target_y, dtype=float)
    source_values = np.asarray(source_values, dtype=float)

    values_at_target_x = np.empty((len(target_x), len(source_y)), dtype=float)
    for j in range(len(source_y)):
        values_at_target_x[:, j] = np.interp(target_x, source_x, source_values[:, j])

    values_at_target = np.empty((len(target_x), len(target_y)), dtype=float)
    for i in range(len(target_x)):
        values_at_target[i, :] = np.interp(target_y, source_y, values_at_target_x[i, :])
    return values_at_target


def _calculate_deficit_calibration(dynamic_flow_field, foxes_flow_field, threshold_percent=0.2):
    dynamic_deficit = _flow_deficit_percent(dynamic_flow_field)
    foxes_deficit = _flow_deficit_percent(foxes_flow_field)
    foxes_on_dynamic = _interpolate_grid_to_target(
        foxes_flow_field["x"],
        foxes_flow_field["y"],
        foxes_deficit,
        dynamic_flow_field["x"],
        dynamic_flow_field["y"],
    )

    mask = (
        np.isfinite(dynamic_deficit)
        & np.isfinite(foxes_on_dynamic)
        & ((dynamic_deficit >= threshold_percent) | (foxes_on_dynamic >= threshold_percent))
    )
    if int(mask.sum()) < 20:
        return None

    dyn = dynamic_deficit[mask]
    ref = foxes_on_dynamic[mask]
    denom = float(np.sum(dyn * dyn))
    if denom <= 1e-12:
        return None

    factor = float(np.sum(dyn * ref) / denom)
    factor = float(np.clip(factor, 0.2, 5.0))
    rmse_before = float(np.sqrt(np.mean((dyn - ref) ** 2)))
    rmse_after = float(np.sqrt(np.mean((factor * dyn - ref) ** 2)))
    return {
        "factor": factor,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "cells": int(mask.sum()),
        "mean_dynamic_deficit": float(np.mean(dyn)),
        "mean_foxes_deficit": float(np.mean(ref)),
    }


def _wake_extent_m(flow_field, layout_df, wind_direction, threshold_percent=2.0):
    deficit = _flow_deficit_percent(flow_field)
    x_values = np.asarray(flow_field["x"], dtype=float)
    y_values = np.asarray(flow_field["y"], dtype=float)
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    if layout_df is None or layout_df.empty:
        origin_x = float(x_values[0])
        origin_y = 0.0
    else:
        origin_x = float(layout_df.iloc[0]["x"])
        origin_y = float(layout_df.iloc[0]["y"])

    wd_rad = np.deg2rad(float(wind_direction))
    downstream = np.array([-np.sin(wd_rad), -np.cos(wd_rad)], dtype=float)
    projection = (xx - origin_x) * downstream[0] + (yy - origin_y) * downstream[1]
    mask = np.isfinite(deficit) & (deficit >= threshold_percent) & (projection > 0.0)
    if not np.any(mask):
        return 0.0
    return float(np.nanmax(projection[mask]))


def _calculate_wake_length_calibration(
    dynamic_flow_field,
    dynamic_layout_df,
    foxes_flow_field,
    foxes_layout_df,
    wind_direction,
    current_decay_length_d,
    threshold_percent=2.0,
):
    dynamic_length = _wake_extent_m(
        dynamic_flow_field,
        dynamic_layout_df,
        wind_direction,
        threshold_percent=threshold_percent,
    )
    foxes_length = _wake_extent_m(
        foxes_flow_field,
        foxes_layout_df,
        wind_direction,
        threshold_percent=threshold_percent,
    )
    if dynamic_length <= 1e-9 or foxes_length <= 1e-9:
        return None

    factor = float(np.clip(foxes_length / dynamic_length, 0.25, 4.0))
    new_decay_length_d = float(np.clip(current_decay_length_d * factor, 5.0, 250.0))
    return {
        "factor": factor,
        "old_decay_length_d": float(current_decay_length_d),
        "new_decay_length_d": new_decay_length_d,
        "dynamic_length_m": dynamic_length,
        "foxes_length_m": foxes_length,
        "threshold_percent": float(threshold_percent),
    }


def _get_foxes_reference_cache():
    cache = st.session_state.setdefault("dynamic_foxes_reference_cache", {})
    order = st.session_state.setdefault("dynamic_foxes_reference_cache_order", [])
    return cache, order


def _store_foxes_reference(signature, result, max_entries=12):
    cache, order = _get_foxes_reference_cache()
    cache[signature] = result
    if signature in order:
        order.remove(signature)
    order.append(signature)
    while len(order) > max_entries:
        oldest = order.pop(0)
        cache.pop(oldest, None)


def _load_foxes_reference(signature):
    cache, order = _get_foxes_reference_cache()
    if signature not in cache:
        return None
    if signature in order:
        order.remove(signature)
    order.append(signature)
    return cache[signature]


def _clear_foxes_reference_cache():
    st.session_state.dynamic_foxes_reference_cache = {}
    st.session_state.dynamic_foxes_reference_cache_order = []
    st.session_state.dynamic_foxes_reference = None


def _runner_reference_signature(config):
    return (
        config.layout_mode,
        config.n_turbines,
        float(config.spacing_x),
        float(config.spacing_y),
        tuple(float(v) for v in config.hub_heights),
        float(config.plane_height),
        float(config.x_min),
        float(config.x_max),
        float(config.y_min),
        float(config.y_max),
        float(config.wind_speed),
        float(config.wind_direction),
        float(config.turbulence_intensity),
        float(config.air_density),
        config.atmosphere_preset,
        config.vertical_profile,
        float(config.shear_exponent),
        float(config.roughness_length),
        float(config.monin_obukhov_length),
        config.turbine_label,
        config.wake_label,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render_dynamic_runner(base_config):
    st.info(
        "Experimenteller dynamischer Runner: Wake-Pakete werden transportiert, "
        "Boeenfronten laufen durch das Gebiet, und Turbinenleistungen reagieren auf den lokalen effektiven Zufluss."
    )

    controls = st.columns(5)
    if controls[0].button("Start", type="primary", use_container_width=True, key="dynamic_start"):
        st.session_state.dynamic_runner_active = True
        st.rerun()
    if controls[1].button("Pause", use_container_width=True, key="dynamic_pause"):
        st.session_state.dynamic_runner_active = False
    if controls[2].button("Einzelschritt", use_container_width=True, key="dynamic_step"):
        st.session_state.dynamic_runner_do_single_step = True
    if controls[3].button("Stop", use_container_width=True, key="dynamic_stop"):
        st.session_state.dynamic_runner_active = False
    if controls[4].button("Reset", use_container_width=True, key="dynamic_reset"):
        st.session_state.dynamic_runner_active = False
        st.session_state.dynamic_runner_state = None
        st.session_state.dynamic_runner_effective_config = None
        st.session_state.dynamic_runner_error = None
        st.session_state.dynamic_foxes_reference = None
        st.session_state.dynamic_deficit_calibration = 1.0
        st.session_state.dynamic_wake_decay_length_d = 45.0
        st.session_state.pending_dynamic_wake_decay_length_d = 45.0
        st.session_state.dynamic_calibration_info = None
        st.session_state.dynamic_wake_length_calibration_info = None
        st.session_state.dynamic_foxes_reference_cache_status = None

    live = st.session_state.get("dynamic_runner_live_config", base_config)
    if "pending_dynamic_wake_decay_length_d" in st.session_state:
        st.session_state.dyn_wake_decay_length_d_slider = st.session_state.pop(
            "pending_dynamic_wake_decay_length_d"
        )
    left, right = st.columns([0.82, 1.18])
    with left:
        st.subheader("Live-Parameter")
        block_minutes = st.number_input("Blockgroesse [min]", 0.1, 60.0, 0.5, 0.1, key="dyn_block_minutes")
        st.caption("Kleinere Werte zeigen Boeen und Wake-Pakete fluessiger, erhoehen aber den Darstellungsaufwand.")
        auto_delay_s = st.number_input("Anzeige-Takt [s]", 0.2, 10.0, 0.8, 0.2, key="dyn_auto_delay_s")
        live_ws = st.slider("Windgeschwindigkeit [m/s]", 3.0, 25.0, float(live.wind_speed), 0.5, key="dyn_live_ws")
        live_wd = st.slider("Windrichtung [deg]", 0.0, 360.0, float(live.wind_direction), 1.0, key="dyn_live_wd")
        live_ti = st.slider(
            "Turbulenzintensitaet [-]",
            0.01,
            0.30,
            float(live.turbulence_intensity),
            0.01,
            key="dyn_live_ti",
        )
        live_rho = st.number_input("Luftdichte [kg/m3]", 0.9, 1.4, float(live.air_density), 0.005, key="dyn_live_rho")

        st.markdown("**Heatmap-Skalierung**")
        heatmap_scale_mode = st.radio(
            "Skalierungsmodus",
            ["Automatisch", "Manuell"],
            horizontal=True,
            key="dynamic_heatmap_scale_mode",
        )
        heatmap_ws_max = st.number_input(
            "WS max Farbskala [m/s]",
            min_value=1.0,
            max_value=60.0,
            value=25.0,
            step=0.5,
            disabled=heatmap_scale_mode == "Automatisch",
            key="dynamic_heatmap_ws_max",
        )

        st.markdown("**Aenderungsrampen**")
        ws_rate_per_h = st.slider("Max. dWS/dt [m/s pro h]", 0.5, 30.0, 6.0, 0.5, key="dyn_ws_rate")
        wd_rate_per_h = st.slider("Max. dWD/dt [deg pro h]", 5.0, 240.0, 60.0, 5.0, key="dyn_wd_rate")
        ti_rate_per_h = st.slider("Max. dTI/dt [1 pro h]", 0.005, 0.300, 0.060, 0.005, key="dyn_ti_rate")

        st.markdown("**Wake-Pakete**")
        release_interval_s = st.slider("Paket-Intervall [s]", 1.0, 120.0, 5.0, 1.0, key="dyn_release_interval")
        packet_lifetime_s = st.slider("Paket-Lebensdauer [s]", 120.0, 3600.0, 1200.0, 60.0, key="dyn_lifetime")
        wake_decay_length_d = st.slider(
            "Wake-Abklinglaenge [D]",
            5.0,
            250.0,
            float(st.session_state.get("dynamic_wake_decay_length_d", 45.0)),
            5.0,
            key="dyn_wake_decay_length_d_slider",
        )
        st.session_state.dynamic_wake_decay_length_d = wake_decay_length_d
        st.caption("Physikalische Defizit-Abschwaechung entlang der Stroemung; Paket-Lebensdauer bleibt technische Obergrenze.")
        wake_width0_d = st.slider("Startbreite [D]", 0.10, 1.50, 0.35, 0.05, key="dyn_width0")
        wake_growth = st.slider("Wake-Aufweitung [-]", 0.005, 0.120, 0.035, 0.005, key="dyn_growth")
        manual_deficit_gain = st.slider("Defizit-Staerke [-]", 0.20, 2.00, 1.00, 0.05, key="dyn_deficit_gain")
        calibration_factor = float(st.session_state.get("dynamic_deficit_calibration", 1.0))
        st.caption(
            f"Wirksame Defizit-Staerke: {manual_deficit_gain * calibration_factor:.2f} "
            f"(FOXES-Faktor {calibration_factor:.2f})"
        )
        superposition = st.selectbox(
            "Wake-Superposition",
            ["quadratisch", "linear begrenzt", "maximal", "FOXES-nah"],
            index=3,
            key="dyn_superposition",
        )
        st.caption(
            "Quadratisch entspricht dem bisherigen Energie-Defizit-Ansatz; linear addiert staerker, "
            "maximal nutzt nur das dominante Paket."
        )
        rotor_sampling_points = st.selectbox(
            "Rotor-Sampling",
            [1, 5, 9, 15, 21],
            index=2,
            format_func=lambda n: "Nabenpunkt" if n == 1 else f"{n} Punkte ueber Rotorbreite",
            key="dyn_rotor_sampling_points",
        )
        st.caption("2D-Ersatz fuer REWS: gewichtete Stützpunkte quer zur lokalen Windrichtung.")
        show_dynamic_packet_paths = st.checkbox(
            "Wake-Paketbahnen anzeigen",
            value=False,
            key="dyn_show_packet_paths",
        )
        meander_scale = st.slider("Maandrier-Staerke [-]", 0.0, 3.0, 0.75, 0.05, key="dyn_meander")
        max_packets = st.slider("Max. Pakete", 100, 3000, 1200, 100, key="dyn_max_packets")

        st.markdown("**Boeenfront**")
        gust_delta_ws = st.slider("Boeen-dWS [m/s]", -8.0, 8.0, 3.0, 0.5, key="dyn_gust_dws")
        gust_delta_wd = st.slider("Boeen-dWD [deg]", -45.0, 45.0, 0.0, 1.0, key="dyn_gust_dwd")
        gust_speed = st.slider("Frontgeschwindigkeit [m/s]", 1.0, 35.0, 10.0, 0.5, key="dyn_gust_speed")
        gust_width = st.slider("Frontbreite [m]", 50.0, 2000.0, 300.0, 50.0, key="dyn_gust_width")

    params = DynamicRunnerParameters(
        release_interval_s=release_interval_s,
        packet_lifetime_s=packet_lifetime_s,
        wake_decay_length_d=wake_decay_length_d,
        wake_width0_d=wake_width0_d,
        wake_growth=wake_growth,
        deficit_gain=manual_deficit_gain * calibration_factor,
        superposition="quadratisch" if superposition == "FOXES-nah" else superposition,
        rotor_sampling_points=rotor_sampling_points,
        meander_scale=meander_scale,
        max_packets=max_packets,
        gust_delta_ws=gust_delta_ws,
        gust_delta_wd=gust_delta_wd,
        gust_speed=gust_speed,
        gust_width=gust_width,
    )
    live_config = replace(
        base_config,
        wind_speed=live_ws,
        wind_direction=live_wd,
        turbulence_intensity=live_ti,
        air_density=live_rho,
        particle_tracking=False,
        grid_points=min(base_config.grid_points, 75),
    )
    st.session_state.dynamic_runner_live_config = live_config
    effective_config = st.session_state.get("dynamic_runner_effective_config", live_config) or live_config
    # Die FOXES-Referenz wird im Dynamic Runner nie mit Streamline-/
    # Kontur-Wake-Laenge angezeigt (siehe _wake_field_figure-Aufrufe unten,
    # show_streamline_wake_length/show_contour_wake_length stehen dort immer
    # auf False) -> die entsprechenden teuren Berechnungen in run_single_state
    # von vornherein abschalten.
    foxes_reference_config = replace(
        effective_config,
        particle_tracking=False,
        grid_points=min(base_config.grid_points, 55),
        compute_streamline_wake_length=False,
        compute_contour_wake_length=False,
    )
    foxes_reference_signature = _runner_reference_signature(foxes_reference_config)

    if st.session_state.get("dynamic_runner_state") is None:
        try:
            st.session_state.dynamic_runner_effective_config = live_config
            effective_config = live_config
            st.session_state.dynamic_runner_state = create_dynamic_runner_state(effective_config, params)
            st.session_state.dynamic_runner_error = None
        except Exception as exc:
            st.session_state.dynamic_runner_error = str(exc)

    with left:
        if st.button("Boeenfront ausloesen", use_container_width=True, key="dynamic_add_gust"):
            try:
                add_gust_front(st.session_state.dynamic_runner_state, effective_config, params)
                st.session_state.dynamic_runner_state["last_result"] = evaluate_dynamic_state(
                    st.session_state.dynamic_runner_state,
                    effective_config,
                    params,
                )
                st.session_state.dynamic_runner_error = None
            except Exception as exc:
                st.session_state.dynamic_runner_error = str(exc)
        st.markdown("**FOXES-Referenz**")
        cache, _cache_order = _get_foxes_reference_cache()
        cache_hit_available = foxes_reference_signature in cache
        if st.button("FOXES-Referenz aktualisieren", use_container_width=True, key="dynamic_update_foxes_reference"):
            try:
                cached_reference = _load_foxes_reference(foxes_reference_signature)
                if cached_reference is None:
                    with st.spinner("FOXES berechnet stationaeres Referenzfeld..."):
                        cached_reference = run_single_state(foxes_reference_config)
                    _store_foxes_reference(foxes_reference_signature, cached_reference)
                    st.session_state.dynamic_foxes_reference_cache_status = "neu berechnet"
                else:
                    st.session_state.dynamic_foxes_reference_cache_status = "aus Cache geladen"
                st.session_state.dynamic_foxes_reference = (
                    foxes_reference_signature,
                    cached_reference,
                )
                st.session_state.dynamic_runner_error = None
            except Exception as exc:
                st.session_state.dynamic_runner_error = str(exc)
        cache_cols = st.columns(2)
        cache_cols[0].caption(
            f"Cache: {len(cache)} Eintraege"
            + (" | Treffer verfuegbar" if cache_hit_available else "")
        )
        if cache_cols[1].button("Cache leeren", use_container_width=True, key="dynamic_clear_foxes_cache"):
            _clear_foxes_reference_cache()
            st.session_state.dynamic_foxes_reference_cache_status = "geleert"
            st.rerun()
        if st.session_state.get("dynamic_foxes_reference_cache_status"):
            st.caption(f"Letzte Referenz: {st.session_state.dynamic_foxes_reference_cache_status}")
        reference_available = (
            st.session_state.get("dynamic_foxes_reference") is not None
            and st.session_state.dynamic_foxes_reference[0] == foxes_reference_signature
        )
        if reference_available:
            st.caption("Stationaere FOXES-Referenz passt zum aktuellen wirksamen Zustand.")
        elif st.session_state.get("dynamic_foxes_reference") is not None:
            st.caption("FOXES-Referenz ist veraltet. Bitte aktualisieren.")
        else:
            st.caption("Noch keine stationaere FOXES-Referenz berechnet.")
        can_calibrate = (
            reference_available
            and st.session_state.get("dynamic_runner_state") is not None
            and st.session_state.dynamic_runner_state.get("last_result") is not None
        )
        if st.button(
            "Paketstaerke gegen FOXES kalibrieren",
            use_container_width=True,
            key="dynamic_calibrate_deficit",
            disabled=not can_calibrate,
        ):
            try:
                _dyn_summary, _dyn_results_df, _dyn_layout_df, dyn_flow_field = st.session_state.dynamic_runner_state[
                    "last_result"
                ]
                _foxes_summary, _foxes_results_df, _foxes_layout_df, foxes_flow_field = (
                    st.session_state.dynamic_foxes_reference[1]
                )
                calibration = _calculate_deficit_calibration(dyn_flow_field, foxes_flow_field)
                if calibration is None:
                    st.session_state.dynamic_calibration_info = {
                        "status": "fehlgeschlagen",
                        "message": "Zu wenig ueberlappende Wake-Defizitflaeche. Runner erst einige Minuten einschwingen lassen.",
                    }
                else:
                    old_factor = float(st.session_state.get("dynamic_deficit_calibration", 1.0))
                    new_factor = float(np.clip(old_factor * calibration["factor"], 0.05, 10.0))
                    st.session_state.dynamic_deficit_calibration = new_factor
                    st.session_state.dynamic_calibration_info = {
                        "status": "ok",
                        "applied_factor": calibration["factor"],
                        "total_factor": new_factor,
                        "rmse_before": calibration["rmse_before"],
                        "rmse_after": calibration["rmse_after"],
                        "cells": calibration["cells"],
                    }
                    st.session_state.dynamic_runner_active = False
                    st.session_state.dynamic_runner_state = None
                st.session_state.dynamic_runner_error = None
                st.rerun()
            except Exception as exc:
                st.session_state.dynamic_runner_error = str(exc)
        if st.button(
            "Wake-Laenge gegen FOXES kalibrieren",
            use_container_width=True,
            key="dynamic_calibrate_wake_length",
            disabled=not can_calibrate,
        ):
            try:
                _dyn_summary, _dyn_results_df, dyn_layout_df, dyn_flow_field = st.session_state.dynamic_runner_state[
                    "last_result"
                ]
                _foxes_summary, _foxes_results_df, foxes_layout_df, foxes_flow_field = (
                    st.session_state.dynamic_foxes_reference[1]
                )
                length_calibration = _calculate_wake_length_calibration(
                    dyn_flow_field,
                    dyn_layout_df,
                    foxes_flow_field,
                    foxes_layout_df,
                    effective_config.wind_direction,
                    wake_decay_length_d,
                    threshold_percent=base_config.wake_length_threshold,
                )
                if length_calibration is None:
                    st.session_state.dynamic_wake_length_calibration_info = {
                        "status": "fehlgeschlagen",
                        "message": "Wake-Laenge konnte nicht bestimmt werden. Defizitschwelle oder Ausschnitt pruefen.",
                    }
                else:
                    st.session_state.dynamic_wake_decay_length_d = length_calibration["new_decay_length_d"]
                    st.session_state.pending_dynamic_wake_decay_length_d = length_calibration["new_decay_length_d"]
                    st.session_state.dynamic_wake_length_calibration_info = {
                        "status": "ok",
                        **length_calibration,
                    }
                    st.session_state.dynamic_runner_active = False
                    st.session_state.dynamic_runner_state = None
                st.session_state.dynamic_runner_error = None
                st.rerun()
            except Exception as exc:
                st.session_state.dynamic_runner_error = str(exc)
        calibration_info = st.session_state.get("dynamic_calibration_info")
        if calibration_info:
            if calibration_info.get("status") == "ok":
                st.caption(
                    f"Kalibriert: Faktor {calibration_info['total_factor']:.2f}, "
                    f"RMSE {calibration_info['rmse_before']:.2f} -> {calibration_info['rmse_after']:.2f} %-Pkt"
                )
            else:
                st.caption(calibration_info.get("message", "Kalibrierung nicht moeglich."))
        wake_length_calibration_info = st.session_state.get("dynamic_wake_length_calibration_info")
        if wake_length_calibration_info:
            if wake_length_calibration_info.get("status") == "ok":
                st.caption(
                    f"Wake-Laenge kalibriert: {wake_length_calibration_info['old_decay_length_d']:.0f}D -> "
                    f"{wake_length_calibration_info['new_decay_length_d']:.0f}D; "
                    f"Dyn {wake_length_calibration_info['dynamic_length_m']:.0f} m, "
                    f"FOXES {wake_length_calibration_info['foxes_length_m']:.0f} m"
                )
            else:
                st.caption(wake_length_calibration_info.get("message", "Wake-Laengen-Kalibrierung nicht moeglich."))
        show_foxes_reference_field = st.checkbox(
            "FOXES-Referenzfeld anzeigen",
            value=reference_available,
            key="dynamic_show_foxes_reference_field",
        )

    def _tick():
        """Der sich wiederholende Teil: Zeitschritt (falls aktiv) + Anzeige.

        Laeuft als st.fragment periodisch fuer sich alleine, ohne dass der
        Rest des Skripts (Sidebar, Regler oben) mit ausgefuehrt wird.
        """
        nonlocal effective_config

        do_step = st.session_state.pop("dynamic_runner_do_single_step", False) or st.session_state.get(
            "dynamic_runner_active", False
        )
        if do_step and st.session_state.get("dynamic_runner_state") is not None:
            try:
                dt_h = block_minutes / 60.0
                effective_config = replace(
                    live_config,
                    wind_speed=float(
                        _move_towards(effective_config.wind_speed, live_config.wind_speed, ws_rate_per_h * dt_h)
                    ),
                    wind_direction=float(
                        _move_angle_towards(
                            effective_config.wind_direction,
                            live_config.wind_direction,
                            wd_rate_per_h * dt_h,
                        )
                    ),
                    turbulence_intensity=float(
                        _move_towards(
                            effective_config.turbulence_intensity,
                            live_config.turbulence_intensity,
                            ti_rate_per_h * dt_h,
                        )
                    ),
                )
                st.session_state.dynamic_runner_state = step_dynamic_runner(
                    st.session_state.dynamic_runner_state,
                    effective_config,
                    params,
                    block_minutes * 60.0,
                )
                st.session_state.dynamic_runner_effective_config = effective_config
                st.session_state.dynamic_runner_error = None
            except Exception as exc:
                st.session_state.dynamic_runner_active = False
                st.session_state.dynamic_runner_error = str(exc)

        if st.session_state.get("dynamic_runner_error"):
            st.error(st.session_state.dynamic_runner_error)
            return

        state = st.session_state.get("dynamic_runner_state")
        if not state or state.get("last_result") is None:
            return

        summary, results_df, layout_df, flow_field = state["last_result"]
        foxes_reference = st.session_state.get("dynamic_foxes_reference")
        foxes_reference_result = None
        foxes_reference_is_current = False
        if foxes_reference is not None:
            foxes_reference_is_current = foxes_reference[0] == foxes_reference_signature
            foxes_reference_result = foxes_reference[1]
        history_df = dynamic_history_dataframe(state)
        total_time_h = float(state.get("time_s", 0.0)) / 3600.0
        total_energy_mwh = float(state.get("energy_mwh", 0.0))
        last_power = summary["power_mw"]
        mean_power = last_power if history_df.empty else float(history_df["power_mw"].mean())
        last_day = history_df[history_df["time_h"] >= max(total_time_h - 24.0, 0.0)] if not history_df.empty else history_df
        last_day_energy = 0.0 if last_day.empty else float(last_day["energy_mwh"].sum())

        with right:
            st.subheader("Statistik")
            mcols = st.columns(4)
            mcols[0].metric("Simulationszeit", f"{total_time_h:.2f} h")
            mcols[1].metric("Aktuelle Leistung", f"{last_power:.2f} MW")
            mcols[2].metric("Mittlere Leistung", f"{mean_power:.2f} MW")
            mcols[3].metric("Ertrag seit Start", f"{total_energy_mwh:.2f} MWh")
            mcols = st.columns(4)
            mcols[0].metric("Ertrag letzte 24 h", f"{last_day_energy:.2f} MWh")
            mcols[1].metric("Wake-Verlust", f"{summary['wake_loss_percent']:.1f} %")
            mcols[2].metric("Wake-Pakete", f"{summary['packet_count']}")
            mcols[3].metric("Boeenfronten", f"{summary['gust_count']}")
            mcols = st.columns(3)
            mcols[0].metric("Mittlere WS im Feld", f"{summary['mean_ws']:.2f} m/s")
            mcols[1].metric("Wirksame WS", f"{effective_config.wind_speed:.2f} m/s", f"Ziel {live_ws:.2f}")
            mcols[2].metric("Wirksame WD", f"{effective_config.wind_direction:.0f} deg", f"Ziel {live_wd:.0f}")
            mcols = st.columns(3)
            mcols[0].metric(
                "Rotor-Sampling", "Nabenpunkt" if rotor_sampling_points == 1 else f"{rotor_sampling_points} Punkte"
            )
            mcols[1].metric("Superposition", superposition)
            mcols[2].metric("FOXES-Faktor", f"{calibration_factor:.2f}")
            mcols = st.columns(3)
            mcols[0].metric("Wake-Abklinglaenge", f"{wake_decay_length_d:.0f} D")
            mcols[1].metric("Paket-Lebensdauer", f"{packet_lifetime_s:.0f} s")
            mcols[2].metric("Paket-Intervall", f"{release_interval_s:.0f} s")
            st.markdown("**Vergleich zur FOXES-Referenz**")
            if foxes_reference_result is None:
                mcols = st.columns(3)
                mcols[0].metric("FOXES stationaer", "nicht berechnet")
                mcols[1].metric("Abweichung dyn.", "-")
                mcols[2].metric("Referenzstatus", "fehlt")
                st.caption("Links im Bereich FOXES-Referenz zuerst 'FOXES-Referenz aktualisieren' ausfuehren.")
            else:
                foxes_summary = foxes_reference_result[0]
                power_delta_percent = (
                    0.0
                    if abs(foxes_summary["power_mw"]) < 1e-12
                    else (summary["power_mw"] - foxes_summary["power_mw"]) / foxes_summary["power_mw"] * 100.0
                )
                wake_delta = summary["wake_loss_percent"] - (
                    100.0 - foxes_summary.get("efficiency_percent", 100.0)
                )
                mcols = st.columns(3)
                mcols[0].metric("FOXES stationaer", f"{foxes_summary['power_mw']:.2f} MW")
                mcols[1].metric("Abweichung dyn.", f"{power_delta_percent:+.1f} %")
                mcols[2].metric("Referenzstatus", "gueltig" if foxes_reference_is_current else "veraltet")
                mcols = st.columns(2)
                mcols[0].metric("FOXES Wake-Verlust", f"{100.0 - foxes_summary.get('efficiency_percent', 100.0):.1f} %")
                mcols[1].metric("Wake-Verlust Delta", f"{wake_delta:+.1f} %-Pkt")
                if not foxes_reference_is_current:
                    st.caption("Die angezeigte FOXES-Referenz passt nicht mehr exakt zum wirksamen Runner-Zustand.")
            calibration_info = st.session_state.get("dynamic_calibration_info")
            if calibration_info and calibration_info.get("status") == "ok":
                mcols = st.columns(3)
                mcols[0].metric("FOXES-Kalibrierfaktor", f"{calibration_info['total_factor']:.2f}")
                mcols[1].metric("Defizit-RMSE vorher", f"{calibration_info['rmse_before']:.2f} %-Pkt")
                mcols[2].metric("Defizit-RMSE nachher", f"{calibration_info['rmse_after']:.2f} %-Pkt")
            wake_length_calibration_info = st.session_state.get("dynamic_wake_length_calibration_info")
            if wake_length_calibration_info and wake_length_calibration_info.get("status") == "ok":
                mcols = st.columns(3)
                mcols[0].metric("Dyn. Wake-Laenge", f"{wake_length_calibration_info['dynamic_length_m']:.0f} m")
                mcols[1].metric("FOXES Wake-Laenge", f"{wake_length_calibration_info['foxes_length_m']:.0f} m")
                mcols[2].metric("Abklinglaenge neu", f"{wake_length_calibration_info['new_decay_length_d']:.0f} D")
            st.caption(
                f"Aktive dynamische Wake-Superposition: {superposition}; "
                f"Rotor-Sampling: {'Nabenpunkt' if rotor_sampling_points == 1 else str(rotor_sampling_points) + ' Punkte'}"
            )

        plot_left, plot_right = st.columns([1.25, 0.95])
        with plot_left:
            st.subheader("Leistungsverlauf")
            st.plotly_chart(_runner_power_figure(history_df), use_container_width=True)
        with plot_right:
            st.subheader("Wake-Verlust")
            st.plotly_chart(_runner_wake_loss_figure(history_df), use_container_width=True)

        st.subheader("Aktuelles dynamisches Windfeld")
        if foxes_reference_result is not None and show_foxes_reference_field:
            _foxes_summary, _foxes_results_df, foxes_layout_df, foxes_flow_field = foxes_reference_result
            # Gemeinsame/verknuepfte Achsen statt zweier unabhaengiger Plots,
            # damit Zoom/Pan in einem Panel synchron im anderen mitlaeuft.
            st.plotly_chart(
                _wake_field_comparison_figure(
                    flow_field,
                    layout_df,
                    "Dynamischer Paket-Layer",
                    foxes_flow_field,
                    foxes_layout_df,
                    "Stationaere FOXES-Referenz" + ("" if foxes_reference_is_current else " (veraltet)"),
                    show_wind_arrows=True,
                    show_particles=show_dynamic_packet_paths,
                    heatmap_scale_mode=heatmap_scale_mode,
                    heatmap_ws_max=heatmap_ws_max,
                    height=560,
                ),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                _wake_field_figure(
                    flow_field,
                    layout_df,
                    show_wind_arrows=True,
                    show_midline_wake_length=False,
                    show_streamline_wake_length=False,
                    show_contour_wake_length=False,
                    show_particles=show_dynamic_packet_paths,
                    heatmap_scale_mode=heatmap_scale_mode,
                    heatmap_ws_max=heatmap_ws_max,
                    height=650,
                ),
                use_container_width=True,
            )
        st.subheader("Aktuelle Turbinenwerte")
        st.dataframe(results_df, use_container_width=True, hide_index=True, height=240)

    run_every = auto_delay_s if st.session_state.get("dynamic_runner_active", False) else None
    st.fragment(run_every=run_every)(_tick)()
