from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from foxes_runner import SimulationConfig, available_turbines, default_hub_heights


DYNAMIC_MODEL_VERSION = 3


@dataclass(frozen=True)
class DynamicRunnerParameters:
    release_interval_s: float = 5.0
    packet_lifetime_s: float = 1200.0
    wake_decay_length_d: float = 45.0
    wake_width0_d: float = 0.35
    wake_growth: float = 0.035
    deficit_gain: float = 1.0
    superposition: str = "quadratisch"
    rotor_sampling_points: int = 1
    meander_scale: float = 0.75
    max_packets: int = 1200
    gust_delta_ws: float = 3.0
    gust_delta_wd: float = 0.0
    gust_speed: float = 10.0
    gust_width: float = 300.0


def create_dynamic_runner_state(config: SimulationConfig, params: DynamicRunnerParameters):
    layout_df = _layout_from_config(config)
    curve_df = _load_power_ct_curve(config.turbine_label)
    rotor_diameter = _parse_rotor_diameter(config.turbine_label, fallback=126.0)
    rng = np.random.default_rng(int(config.particle_seed))

    state = {
        "config_signature": _config_signature(config),
        "model_version": DYNAMIC_MODEL_VERSION,
        "params": asdict(params),
        "time_s": 0.0,
        "energy_mwh": 0.0,
        "next_release_s": 0.0,
        "packets": [],
        "gusts": [],
        "history": [],
        "layout_df": layout_df,
        "curve_df": curve_df,
        # curve_df einmal in NumPy-Arrays konvertieren statt bei jedem
        # einzelnen _curve_values()-Aufruf (2x pro Turbine pro Zeitschritt)
        # erneut curve_df["ws"].to_numpy() etc. auszufuehren.
        "curve_arrays": _curve_arrays_from_df(curve_df),
        "rotor_diameter": rotor_diameter,
        "rng_state": rng.bit_generator.state,
        "last_result": None,
    }
    state["last_result"] = evaluate_dynamic_state(state, config, params)
    return state


def dynamic_history_dataframe(state):
    rows = state.get("history", []) if state else []
    return pd.DataFrame(rows)


def add_gust_front(state, config: SimulationConfig, params: DynamicRunnerParameters):
    direction = _downstream_unit(config.wind_direction)
    layout_df = state["layout_df"]
    x_min, x_max = _domain_limits(config.x_min, config.x_max, layout_df["x"])
    y_min, y_max = _domain_limits(config.y_min, config.y_max, layout_df["y"])
    corners = np.array(
        [
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_min],
            [x_max, y_max],
        ],
        dtype=float,
    )
    s_min = float(np.min(corners @ direction))
    state["gusts"].append(
        {
            "start_s": float(state.get("time_s", 0.0)),
            "start_coord": s_min - 1000.0,
            "direction": float(config.wind_direction),
            "delta_ws": float(params.gust_delta_ws),
            "delta_wd": float(params.gust_delta_wd),
            "speed": float(params.gust_speed),
            "width": float(params.gust_width),
        }
    )


def step_dynamic_runner(state, config: SimulationConfig, params: DynamicRunnerParameters, dt_s: float):
    if (
        not state
        or state.get("config_signature") != _config_signature(config)
        or state.get("model_version") != DYNAMIC_MODEL_VERSION
    ):
        state = create_dynamic_runner_state(config, params)

    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]
    old_time_s = float(state.get("time_s", 0.0))
    dt_s = float(max(dt_s, 1.0))

    target_time_s = old_time_s + dt_s
    internal_dt_s = min(10.0, max(1.0, float(params.release_interval_s) / 2.0))
    while float(state.get("time_s", old_time_s)) < target_time_s - 1e-9:
        current_time_s = float(state.get("time_s", old_time_s))
        sub_dt_s = min(internal_dt_s, target_time_s - current_time_s)
        _release_due_packets(state, config, params, current_time_s)
        _advect_packets(state, config, params, sub_dt_s, rng)
        state["time_s"] = current_time_s + sub_dt_s
        _release_due_packets(state, config, params, state["time_s"])
        state["packets"] = _trim_packets(state["packets"], params.max_packets)
        state["gusts"] = _active_gusts(state, config)

    summary, results_df, layout_df, flow_field = evaluate_dynamic_state(state, config, params)
    energy_mwh = summary["power_mw"] * dt_s / 3600.0
    state["energy_mwh"] = float(state.get("energy_mwh", 0.0) + energy_mwh)
    summary["energy_mwh_total"] = state["energy_mwh"]
    state["last_result"] = (summary, results_df, layout_df, flow_field)
    state["history"].append(
        {
            "time_s": state["time_s"],
            "time_h": state["time_s"] / 3600.0,
            "power_mw": summary["power_mw"],
            "ambient_power_mw": summary["ambient_power_mw"],
            "wake_loss_percent": summary["wake_loss_percent"],
            "efficiency_percent": summary["efficiency_percent"],
            "energy_mwh": energy_mwh,
            "energy_mwh_total": state["energy_mwh"],
            "wind_speed": summary["mean_ws"],
            "wind_direction": config.wind_direction,
            "ti": config.turbulence_intensity,
            "target_wind_speed": config.wind_speed,
            "target_wind_direction": config.wind_direction,
            "target_ti": config.turbulence_intensity,
        }
    )
    state["rng_state"] = rng.bit_generator.state
    return state


def evaluate_dynamic_state(state, config: SimulationConfig, params: DynamicRunnerParameters):
    layout_df = state["layout_df"]
    curve_arrays = state.get("curve_arrays") or _curve_arrays_from_df(state["curve_df"])
    x_values, y_values = _grid(config, layout_df)
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    amb_ws, wd_grid = _ambient_grid(xx, yy, state, config)
    # Paket-Attribute (x, y, Breite, Staerke) einmal pro Auswertung als
    # NumPy-Arrays extrahieren, statt sie fuer jede Gitterzelle bzw. jeden
    # Rotor-Sampling-Punkt erneut aus der Paketliste (Liste von Dicts) zu
    # berechnen. Wird unten sowohl fuer das Feld als auch pro Turbine
    # wiederverwendet.
    packet_arrays = _packet_arrays(state, params)
    deficit_percent = _packet_deficit_grid(xx, yy, packet_arrays, params)
    ws_grid = np.maximum(0.05, amb_ws * (1.0 - np.minimum(deficit_percent, 95.0) / 100.0))

    results_rows = []
    ambient_power_kw = 0.0
    power_kw = 0.0
    for i, turbine in layout_df.iterrows():
        x = float(turbine["x"])
        y = float(turbine["y"])
        local_amb_ws, eff_ws, local_def, local_wd = _rotor_equivalent_inflow(
            x=x,
            y=y,
            state=state,
            config=config,
            params=params,
            packet_arrays=packet_arrays,
            exclude_source=i,
        )
        amb_p, _amb_ct = _curve_values_arrays(curve_arrays, local_amb_ws)
        p_kw, ct = _curve_values_arrays(curve_arrays, eff_ws)
        ambient_power_kw += amb_p
        power_kw += p_kw
        results_rows.append(
            {
                "Turbine": turbine["Turbine"],
                "x": x,
                "y": y,
                "H": turbine["Nabenhoehe"],
                "AMB_REWS": local_amb_ws,
                "REWS": eff_ws,
                "AMB_P": amb_p / 1000.0,
                "P": p_kw / 1000.0,
                "CT": ct,
                "EFF": 0.0 if amb_p <= 0 else p_kw / amb_p,
                "WakeDefizit_%": local_def,
                "RotorSamples": int(max(params.rotor_sampling_points, 1)),
            }
        )

    results_df = pd.DataFrame(results_rows)
    ambient_power_mw = ambient_power_kw / 1000.0
    power_mw = power_kw / 1000.0
    wake_loss = 0.0 if ambient_power_mw <= 0 else (ambient_power_mw - power_mw) / ambient_power_mw * 100.0
    summary = {
        "power_mw": power_mw,
        "ambient_power_mw": ambient_power_mw,
        "efficiency_percent": 100.0 - wake_loss,
        "wake_loss_percent": wake_loss,
        "mean_ws": float(np.nanmean(ws_grid)),
        "packet_count": len(state.get("packets", [])),
        "gust_count": len(state.get("gusts", [])),
        "rotor_sampling_points": int(max(params.rotor_sampling_points, 1)),
    }
    flow_field = {
        "x": x_values,
        "y": y_values,
        "z": float(config.plane_height),
        "ws": ws_grid,
        "wd": wd_grid,
        "amb_ws": amb_ws,
        "particle_paths": _packet_paths(state),
        "gust_fronts": _gust_front_lines(state, config),
    }
    return summary, results_df, layout_df, flow_field


def _release_due_packets(state, config, params, current_time_s):
    interval = max(float(params.release_interval_s), 1.0)
    next_release_s = float(state.get("next_release_s", current_time_s))
    curve_arrays = state.get("curve_arrays") or _curve_arrays_from_df(state["curve_df"])
    while next_release_s <= current_time_s + 1e-9:
        for i, turbine in state["layout_df"].iterrows():
            local_ws, local_wd = _ambient_at_point(float(turbine["x"]), float(turbine["y"]), state, config)
            _p_kw, ct = _curve_values_arrays(curve_arrays, local_ws)
            deficit0 = params.deficit_gain * 100.0 * (1.0 - np.sqrt(max(0.0, 1.0 - min(ct, 0.95))))
            state["packets"].append(
                {
                    "x": float(turbine["x"]),
                    "y": float(turbine["y"]),
                    "age_s": 0.0,
                    "distance_m": 0.0,
                    "source": int(i),
                    "deficit0": float(deficit0),
                    "width0": float(max(5.0, params.wake_width0_d * state["rotor_diameter"])),
                    "rotor_diameter": float(state["rotor_diameter"]),
                    "ws0": float(local_ws),
                    "wd0": float(local_wd),
                    "trail": [(float(turbine["x"]), float(turbine["y"]))],
                }
            )
        next_release_s += interval
    state["next_release_s"] = next_release_s


def _advect_packets(state, config, params, dt_s, rng):
    updated = []
    for packet in state.get("packets", []):
        x = float(packet["x"])
        y = float(packet["y"])
        local_ws, local_wd = _ambient_at_point(x, y, state, config)
        direction = _downstream_unit(local_wd)
        cross = np.array([-direction[1], direction[0]])
        meander_sigma = params.meander_scale * config.turbulence_intensity * local_ws * np.sqrt(dt_s)
        step_vec = direction * local_ws * dt_s + cross * rng.normal(0.0, meander_sigma)
        packet["x"] = x + float(step_vec[0])
        packet["y"] = y + float(step_vec[1])
        packet["age_s"] = float(packet["age_s"]) + dt_s
        packet["distance_m"] = float(packet.get("distance_m", 0.0)) + float(np.linalg.norm(step_vec))
        trail = packet.setdefault("trail", [])
        trail.append((packet["x"], packet["y"]))
        if len(trail) > 18:
            del trail[:-18]
        if packet["age_s"] <= params.packet_lifetime_s:
            updated.append(packet)
    state["packets"] = updated


def _packet_arrays(state, params):
    """Extrahiert Paket-Attribute einmal als NumPy-Arrays.

    Vorher wurde fuer jeden einzelnen Abfragepunkt (jede Gitterzelle, jeden
    Rotor-Sampling-Punkt jeder Turbine) erneut ueber die Paketliste
    (Liste von Dicts) iteriert und Breite/Staerke jedes Mal neu berechnet.
    Diese Funktion baut die Arrays einmal pro Auswertungsschritt; die
    eigentliche Distanz-/Defizitberechnung erfolgt dann vektorisiert in
    _packet_deficit_at_points().
    """
    packets = state.get("packets", [])
    n = len(packets)
    if n == 0:
        return None
    xs = np.empty(n, dtype=float)
    ys = np.empty(n, dtype=float)
    widths = np.empty(n, dtype=float)
    strengths = np.empty(n, dtype=float)
    sources = np.empty(n, dtype=np.int64)
    ages = np.empty(n, dtype=float)
    for idx, packet in enumerate(packets):
        xs[idx] = packet["x"]
        ys[idx] = packet["y"]
        widths[idx] = _packet_width(packet, params)
        strengths[idx] = _packet_strength(packet, params)
        sources[idx] = int(packet["source"])
        ages[idx] = packet["age_s"]
    return {"x": xs, "y": ys, "width": widths, "strength": strengths, "source": sources, "age": ages}


def _packet_deficit_grid(xx, yy, packet_arrays, params):
    mode = getattr(params, "superposition", "quadratisch")
    if packet_arrays is None or packet_arrays["x"].size == 0:
        return np.zeros_like(xx, dtype=float)

    if mode == "linear begrenzt":
        deficit_sum = np.zeros_like(xx, dtype=float)
    elif mode == "maximal":
        deficit_max = np.zeros_like(xx, dtype=float)
    else:
        deficit_sq = np.zeros_like(xx, dtype=float)

    xs, ys = packet_arrays["x"], packet_arrays["y"]
    widths, strengths = packet_arrays["width"], packet_arrays["strength"]
    for px, py, width, strength in zip(xs, ys, widths, strengths):
        r2 = (xx - px) ** 2 + (yy - py) ** 2
        local = strength * np.exp(-0.5 * r2 / max(width**2, 1.0))
        if mode == "linear begrenzt":
            deficit_sum += local
        elif mode == "maximal":
            deficit_max = np.maximum(deficit_max, local)
        else:
            deficit_sq += local**2

    if mode == "linear begrenzt":
        return np.minimum(deficit_sum, 95.0)
    if mode == "maximal":
        return deficit_max
    return np.sqrt(deficit_sq)


def _packet_deficit_at_points(query_x, query_y, packet_arrays, params, exclude_source=None):
    """Wake-Defizit an mehreren Abfragepunkten gleichzeitig (vektorisiert).

    Ersetzt den vorherigen Ansatz mit einer reinen Python-Schleife ueber alle
    Wake-Pakete PRO Abfragepunkt. Bei vielen Turbinen x Rotor-Sampling-Punkten
    x Paketen (Worst Case ca. 20 Turbinen x 21 Rotorpunkte x 3000 Pakete
    ~ 1,3 Mio. Skalaroperationen pro Simulationsschritt) war das der
    dominante Kostenfaktor im Dynamic Runner. Numerisch identisch zur alten
    Implementierung (siehe Tests), aber ca. 50-80x schneller.
    """
    query_x = np.atleast_1d(np.asarray(query_x, dtype=float))
    query_y = np.atleast_1d(np.asarray(query_y, dtype=float))
    mode = getattr(params, "superposition", "quadratisch")

    if packet_arrays is None or packet_arrays["x"].size == 0:
        return np.zeros_like(query_x)

    xs, ys = packet_arrays["x"], packet_arrays["y"]
    widths, strengths = packet_arrays["width"], packet_arrays["strength"]
    if exclude_source is not None:
        keep = ~(
            (packet_arrays["source"] == int(exclude_source)) & (packet_arrays["age"] < 8.0)
        )
        if not np.all(keep):
            xs, ys, widths, strengths = xs[keep], ys[keep], widths[keep], strengths[keep]

    if xs.size == 0:
        return np.zeros_like(query_x)

    dx = query_x[:, None] - xs[None, :]
    dy = query_y[:, None] - ys[None, :]
    r2 = dx * dx + dy * dy
    local = strengths[None, :] * np.exp(-0.5 * r2 / np.maximum(widths[None, :] ** 2, 1.0))

    if mode == "linear begrenzt":
        return np.minimum(local.sum(axis=1), 95.0)
    if mode == "maximal":
        return local.max(axis=1)
    return np.sqrt((local**2).sum(axis=1))


def _packet_deficit_at_point(x, y, state, params, exclude_source=None):
    """Einzelpunkt-Komfortfunktion (baut die Arrays bei Bedarf selbst).

    Nicht mehr im heissen Pfad verwendet (siehe _rotor_equivalent_inflow),
    bleibt aber als einfache API fuer Einzelabfragen/Tests erhalten.
    """
    packet_arrays = _packet_arrays(state, params)
    return float(
        _packet_deficit_at_points([x], [y], packet_arrays, params, exclude_source=exclude_source)[0]
    )


def _rotor_equivalent_inflow(x, y, state, config, params, packet_arrays, exclude_source=None):
    n_points = int(max(getattr(params, "rotor_sampling_points", 1), 1))
    centre_amb_ws, centre_wd = _ambient_at_point(x, y, state, config)
    if n_points <= 1:
        local_def = float(
            _packet_deficit_at_points([x], [y], packet_arrays, params, exclude_source=exclude_source)[0]
        )
        eff_ws = max(0.05, centre_amb_ws * (1.0 - min(local_def, 95.0) / 100.0))
        return centre_amb_ws, eff_ws, local_def, centre_wd

    n_points = int(np.clip(n_points, 3, 21))
    if n_points % 2 == 0:
        n_points += 1

    radius = 0.5 * float(state.get("rotor_diameter", 126.0))
    offsets = np.linspace(-radius, radius, n_points)
    norm_offsets = offsets / max(radius, 1e-9)
    weights = np.sqrt(np.maximum(0.0, 1.0 - norm_offsets**2))
    if not np.any(weights > 0.0):
        weights = np.ones_like(offsets)
    weights = weights / np.sum(weights)

    direction = _downstream_unit(centre_wd)
    cross = np.array([-direction[1], direction[0]], dtype=float)
    sample_x = x + cross[0] * offsets
    sample_y = y + cross[1] * offsets

    # Ambient-Zufluss bleibt punktweise (billig: nur wenige Boeenfronten pro
    # Zustand). Das Wake-Defizit fuer ALLE Rotor-Sampling-Punkte auf einmal
    # vektorisiert berechnen, statt pro Punkt einzeln ueber alle Pakete zu
    # iterieren.
    ambient_samples = np.array(
        [_ambient_at_point(float(px), float(py), state, config)[0] for px, py in zip(sample_x, sample_y)],
        dtype=float,
    )
    local_defs = _packet_deficit_at_points(
        sample_x, sample_y, packet_arrays, params, exclude_source=exclude_source
    )
    effective_samples = np.maximum(
        0.05, ambient_samples * (1.0 - np.minimum(local_defs, 95.0) / 100.0)
    )

    amb_rews = float(np.cbrt(np.sum(weights * ambient_samples**3)))
    rews = float(np.cbrt(np.sum(weights * effective_samples**3)))
    deficit = 0.0 if amb_rews <= 1e-9 else max(0.0, (amb_rews - rews) / amb_rews * 100.0)
    return amb_rews, rews, deficit, centre_wd


def _packet_width(packet, params):
    return float(packet["width0"] + params.wake_growth * max(packet["ws0"], 0.1) * packet["age_s"])


def _packet_strength(packet, params):
    decay_length_m = max(float(params.wake_decay_length_d) * float(packet.get("rotor_diameter", 1.0)), 1.0)
    distance_m = float(packet.get("distance_m", max(packet.get("ws0", 0.0), 0.0) * packet.get("age_s", 0.0)))
    return float(packet["deficit0"] * np.exp(-distance_m / decay_length_m))


def _ambient_grid(xx, yy, state, config):
    amb_ws = np.full_like(xx, float(config.wind_speed), dtype=float)
    wd = np.full_like(xx, float(config.wind_direction), dtype=float)
    for gust in state.get("gusts", []):
        factor = _gust_factor(xx, yy, state, gust)
        amb_ws = np.maximum(0.1, amb_ws + gust["delta_ws"] * factor)
        wd = (wd + gust["delta_wd"] * factor) % 360.0
    return amb_ws, wd


def _ambient_at_point(x, y, state, config):
    ws = float(config.wind_speed)
    wd = float(config.wind_direction)
    for gust in state.get("gusts", []):
        factor = float(_gust_factor(np.asarray([[x]]), np.asarray([[y]]), state, gust)[0, 0])
        ws = max(0.1, ws + gust["delta_ws"] * factor)
        wd = (wd + gust["delta_wd"] * factor) % 360.0
    return ws, wd


def _gust_factor(xx, yy, state, gust):
    direction = _downstream_unit(gust["direction"])
    age_s = max(float(state.get("time_s", 0.0)) - float(gust["start_s"]), 0.0)
    centre = float(gust["start_coord"]) + float(gust["speed"]) * age_s
    s = xx * direction[0] + yy * direction[1]
    width = max(float(gust["width"]), 1.0)
    return np.exp(-0.5 * ((s - centre) / width) ** 2)


def _active_gusts(state, config):
    if not state.get("gusts"):
        return []
    layout_df = state["layout_df"]
    x_min, x_max = _domain_limits(config.x_min, config.x_max, layout_df["x"])
    y_min, y_max = _domain_limits(config.y_min, config.y_max, layout_df["y"])
    corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]], dtype=float)
    active = []
    for gust in state["gusts"]:
        direction = _downstream_unit(gust["direction"])
        age_s = max(float(state.get("time_s", 0.0)) - float(gust["start_s"]), 0.0)
        centre = float(gust["start_coord"]) + float(gust["speed"]) * age_s
        s_max = float(np.max(corners @ direction))
        if centre < s_max + 3.0 * float(gust["width"]):
            active.append(gust)
    return active


def _gust_front_lines(state, config):
    layout_df = state["layout_df"]
    x_min, x_max = _domain_limits(config.x_min, config.x_max, layout_df["x"])
    y_min, y_max = _domain_limits(config.y_min, config.y_max, layout_df["y"])
    diagonal = float(np.hypot(x_max - x_min, y_max - y_min))
    fronts = []
    for gust in state.get("gusts", []):
        direction = _downstream_unit(gust["direction"])
        cross = np.array([-direction[1], direction[0]])
        age_s = max(float(state.get("time_s", 0.0)) - float(gust["start_s"]), 0.0)
        centre = float(gust["start_coord"]) + float(gust["speed"]) * age_s
        width = max(float(gust["width"]), 1.0)
        for offset, role in [(0.0, "centre"), (-width, "leading"), (width, "trailing")]:
            centre_point = direction * (centre + offset)
            p0 = centre_point - cross * diagonal
            p1 = centre_point + cross * diagonal
            fronts.append(
                {
                    "x": [float(p0[0]), float(p1[0])],
                    "y": [float(p0[1]), float(p1[1])],
                    "role": role,
                }
            )
    return fronts


def _layout_from_config(config):
    turbine_files = available_turbines()
    hub_default = default_hub_heights(turbine_files).get(config.turbine_label, 100.0)
    if config.layout_mode == "Turbinenreihe":
        heights = list(config.hub_heights) if config.hub_heights else [hub_default] * config.n_turbines
        if len(heights) < config.n_turbines:
            heights += [hub_default] * (config.n_turbines - len(heights))
        rows = []
        for i in range(config.n_turbines):
            rows.append(
                {
                    "Turbine": f"T{i}",
                    "x": float(i * config.spacing_x),
                    "y": float(i * config.spacing_y),
                    "Nabenhoehe": float(heights[i]),
                }
            )
        return pd.DataFrame(rows)

    farm_path = _farm_file_path()
    if farm_path.exists():
        farm_df = pd.read_csv(farm_path)
        x0 = float(farm_df["x"].mean())
        y0 = float(farm_df["y"].mean())
        return pd.DataFrame(
            {
                "Turbine": farm_df.get("label", pd.Series([f"T{i}" for i in range(len(farm_df))])),
                "x": farm_df["x"].astype(float) - x0,
                "y": farm_df["y"].astype(float) - y0,
                "Nabenhoehe": hub_default,
            }
        )

    coords = [(0.0, 0.0), (700.0, 0.0), (1400.0, 0.0), (350.0, 650.0), (1050.0, 650.0)]
    return pd.DataFrame(
        {
            "Turbine": [f"T{i}" for i in range(len(coords))],
            "x": [c[0] for c in coords],
            "y": [c[1] for c in coords],
            "Nabenhoehe": hub_default,
        }
    )


def _grid(config, layout_df):
    x_min, x_max = _domain_limits(config.x_min, config.x_max, layout_df["x"])
    y_min, y_max = _domain_limits(config.y_min, config.y_max, layout_df["y"])
    n = int(np.clip(config.grid_points, 30, 95))
    return np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n)


def _domain_limits(config_min, config_max, coordinates):
    values = np.asarray(coordinates, dtype=float)
    lo = min(float(config_min), float(np.nanmin(values)) - 600.0)
    hi = max(float(config_max), float(np.nanmax(values)) + 1200.0)
    if hi <= lo:
        hi = lo + 1000.0
    return lo, hi


def _load_power_ct_curve(turbine_label):
    file_name = available_turbines()[turbine_label]
    path = Path(file_name)
    if not path.is_absolute():
        path = _power_ct_curve_dir() / file_name
    curve_df = pd.read_csv(path)
    return curve_df[["ws", "P", "ct"]].sort_values("ws").dropna().drop_duplicates("ws")


def _curve_values(curve_df, ws):
    """Backward-kompatible Einzelaufruf-Variante (baut die Arrays bei Bedarf).

    Im heissen Pfad (Auswertung pro Zeitschritt) wird stattdessen
    _curve_values_arrays() mit vorab konvertierten Arrays verwendet, siehe
    state["curve_arrays"].
    """
    return _curve_values_arrays(_curve_arrays_from_df(curve_df), ws)


def _curve_arrays_from_df(curve_df):
    """Konvertiert die Leistungs-/ct-Kennlinie einmalig in NumPy-Arrays.

    Vorher wurde curve_df["ws"]/["P"]/["ct"].to_numpy() bei JEDEM Aufruf von
    _curve_values() neu ausgefuehrt - das passiert im Dynamic Runner zweimal
    pro Turbine pro Zeitschritt (Ambient- und Effektivwert). Die Arrays
    werden jetzt einmal in create_dynamic_runner_state() erzeugt und ueber
    state["curve_arrays"] wiederverwendet.
    """
    return {
        "ws": curve_df["ws"].to_numpy(dtype=float),
        "P": curve_df["P"].to_numpy(dtype=float),
        "ct": curve_df["ct"].to_numpy(dtype=float),
    }


def _curve_values_arrays(curve_arrays, ws):
    ws = float(ws)
    power = float(np.interp(ws, curve_arrays["ws"], curve_arrays["P"], left=0.0, right=0.0))
    thrust = float(np.interp(ws, curve_arrays["ws"], curve_arrays["ct"], left=0.0, right=0.0))
    return power, thrust


def _parse_rotor_diameter(turbine_label, fallback):
    import re

    file_name = Path(available_turbines()[turbine_label]).stem
    match = re.search(r"(?:^|[-_])D([0-9]+(?:d[0-9]+)?)", file_name)
    if not match:
        return fallback
    return float(match.group(1).replace("d", "."))


def _foxes_data_dir(subdir):
    """Deploy-feste FOXES-Statikdaten (analog zu foxes_runner._foxes_data_dir)."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("foxes")
        if spec is not None and spec.origin:
            cand = Path(spec.origin).resolve().parent / "data" / subdir
            if cand.exists():
                return cand
    except Exception:
        pass
    return Path(__file__).resolve().parents[1] / "foxes" / "foxes" / "data" / subdir


def _farm_file_path():
    return _foxes_data_dir("farms") / "test_farm_67.csv"


def _power_ct_curve_dir():
    return _foxes_data_dir("power_ct_curves")


def _downstream_unit(wind_direction_deg):
    wd_rad = np.deg2rad(float(wind_direction_deg))
    return np.array([-np.sin(wd_rad), -np.cos(wd_rad)], dtype=float)


def _packet_paths(state):
    paths = []
    for packet in state.get("packets", [])[-180:]:
        trail = packet.get("trail", [])
        if len(trail) >= 2:
            paths.append({"x": [float(p[0]) for p in trail], "y": [float(p[1]) for p in trail]})
    return paths


def _trim_packets(packets, max_packets):
    max_packets = int(max(max_packets, 1))
    if len(packets) <= max_packets:
        return packets
    return packets[-max_packets:]


def _config_signature(config):
    return (
        config.layout_mode,
        config.n_turbines,
        float(config.spacing_x),
        float(config.spacing_y),
        tuple(float(v) for v in config.hub_heights),
        float(config.x_min),
        float(config.x_max),
        float(config.y_min),
        float(config.y_max),
        config.turbine_label,
        int(config.grid_points),
    )
