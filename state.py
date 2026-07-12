"""Session-State-Verwaltung.

Buendelt an einer Stelle:
- Initialisierung globaler Session-State-Keys
- Das "Ergebnis vorhanden? sonst berechnen"-Muster (ensure_result), das
  vorher fast identisch zweimal in app.py stand (Windklima-Tab und
  Haupt-Simulationstab)
- Reset/Uebergabe-Logik fuer den Runner (_prepare_runner_from_config)
- Die Runner-Verlaufs-DataFrame-Huelle
"""

import pandas as pd
import streamlit as st


def init_app_state():
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "active_workspace" not in st.session_state:
        st.session_state.active_workspace = "Turbine & Windklima"
    if "pending_workspace" in st.session_state:
        st.session_state.active_workspace = st.session_state.pop("pending_workspace")


def ensure_result(
    *,
    config,
    config_key,
    result_key,
    error_key,
    compute_fn,
    run_clicked,
    spinner_text,
    waiting_text,
):
    """Berechnet compute_fn(config) genau dann (neu), wenn noetig, und liefert
    das zwischengespeicherte Ergebnis zurueck.

    Kapselt das Muster: "Button geklickt oder noch kein Ergebnis vorhanden?
    -> Config merken, is_running setzen, rerun -> Berechnung mit
    Fehlerbehandlung -> Fehler anzeigen oder Ergebnis liefern." Vorher fast
    wortgleich zweimal in app.py dupliziert (Windklima-Analyse und
    Turbinenreihe/Windpark-Simulation); jetzt eine einzige Implementierung.

    Beendet das Skript selbst per st.stop(), solange kein Ergebnis vorliegt
    (Fehler oder noch keine erste Berechnung angestossen) - Aufrufer koennen
    also direkt mit dem Rueckgabewert weiterarbeiten.
    """
    if run_clicked:
        st.session_state[config_key] = config
        st.session_state[error_key] = None
        st.session_state.is_running = True
        st.rerun()

    if result_key not in st.session_state and not st.session_state.is_running:
        st.session_state[config_key] = config
        st.session_state[error_key] = None
        st.session_state.is_running = True
        st.rerun()

    if st.session_state.is_running:
        try:
            with st.spinner(spinner_text):
                st.session_state[result_key] = compute_fn(st.session_state[config_key])
                st.session_state[error_key] = None
        except ModuleNotFoundError as exc:
            st.session_state[error_key] = f"Python-Paket fehlt: {exc.name}"
        except Exception as exc:
            st.session_state[error_key] = str(exc)
        finally:
            st.session_state.is_running = False
        st.rerun()

    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])
        st.stop()

    if result_key not in st.session_state:
        st.info(waiting_text)
        st.stop()

    return st.session_state[result_key]


def _runner_history_dataframe():
    columns = [
        "time_s",
        "time_h",
        "power_mw",
        "ambient_power_mw",
        "wake_loss_percent",
        "efficiency_percent",
        "energy_mwh",
        "energy_mwh_total",
        "wind_speed",
        "wind_direction",
        "ti",
        "target_wind_speed",
        "target_wind_direction",
        "target_ti",
    ]
    return pd.DataFrame(st.session_state.get("runner_history", []), columns=columns)


def _prepare_runner_from_config(config, source):
    from dataclasses import replace

    runner_config = replace(
        config,
        particle_tracking=False,
        grid_points=min(config.grid_points, 55),
    )
    st.session_state.runner_source = source
    st.session_state.runner_base_config = runner_config
    st.session_state.runner_live_config = runner_config
    st.session_state.runner_effective_config = runner_config
    st.session_state.runner_active = False
    st.session_state.runner_time_s = 0.0
    st.session_state.runner_energy_mwh = 0.0
    st.session_state.runner_history = []
    st.session_state.runner_last_result = None
    st.session_state.runner_error = None
    st.session_state.dynamic_runner_state = None
    st.session_state.dynamic_runner_active = False
    st.session_state.dynamic_runner_error = None
    st.session_state.dynamic_deficit_calibration = 1.0
    st.session_state.dynamic_wake_decay_length_d = 45.0
    st.session_state.pending_dynamic_wake_decay_length_d = 45.0
    st.session_state.dynamic_calibration_info = None
    st.session_state.dynamic_wake_length_calibration_info = None
    st.session_state.dynamic_foxes_reference = None
    st.session_state.dynamic_foxes_reference_cache_status = None
    st.session_state.pending_workspace = "Runner"
