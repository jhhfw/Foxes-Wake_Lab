from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

# Fuer die schnelle Konturextraktion (siehe _extract_deficit_contours): contourpy
# ist ohnehin eine Abhaengigkeit von matplotlib>=3.8 und wird hier direkt genutzt,
# um den Umweg ueber eine vollstaendige Matplotlib-Figure/Axes zu vermeiden.
import contourpy
from scipy import ndimage


WAKE_MODEL_OPTIONS = {
    "Jensen, k=0.07": ["Jensen_linear_k007"],
    "Bastankhah 2014, k=0.04": ["Bastankhah2014_linear_k004"],
    "Bastankhah + TI-Zunahme": [
        "Bastankhah2014_linear_k004",
        "CrespoHernandez_quadratic_ambka04",
    ],
}

TURBINE_FILES = {
    "NREL 5 MW": "NREL-5MW-D126-H90.csv",
    "IWT 7.5 MW": "IWT-7d5MW-D164-H100.csv",
    "DTU 10 MW": "DTU-10MW-D178d3-H119.csv",
    "IEA 15 MW": "IEA-15MW-D240-H150.csv",
}

DEFAULT_HUB_HEIGHTS = {
    "NREL 5 MW": 90.0,
    "IWT 7.5 MW": 100.0,
    "DTU 10 MW": 119.0,
    "IEA 15 MW": 150.0,
}

CUSTOM_TURBINE_LABEL = "Neue Turbine"


@st.cache_data(show_spinner=False)
def available_turbines():
    """Liste aller verfuegbaren Turbinen (eingebaut + benutzerdefiniert).

    Fuehrt einen Verzeichnis-Scan (glob) durch, der bei jedem Streamlit-Rerun
    unveraendert waere. Wird daher gecacht; nach dem Speichern einer neuen
    Turbine (save_custom_turbine_curve) wird der Cache gezielt invalidiert.
    """
    turbines = dict(TURBINE_FILES)
    for path in _power_ct_curve_dir().glob("*.csv"):
        if path.name not in turbines.values():
            label = path.stem
            turbines[label] = str(path)
    return dict(sorted(turbines.items()))


def default_hub_heights(turbine_files=None):
    turbine_files = turbine_files or available_turbines()
    heights = dict(DEFAULT_HUB_HEIGHTS)
    for label, file_name in turbine_files.items():
        heights.setdefault(label, _parse_hub_height_from_file_name(file_name, 100.0))
    return heights


def save_custom_turbine_curve(config: "WindClimateConfig", curve_df: pd.DataFrame):
    file_name = turbine_file_name(
        config.custom_name,
        config.rated_power_kw,
        config.rotor_diameter,
        config.hub_height,
    )
    target = _power_ct_curve_dir() / file_name
    curve_df[["ws", "P", "ct"]].sort_values("ws").to_csv(target, index=False)
    # Die neue Datei liegt jetzt auf der Platte, der gecachte Verzeichnis-Scan
    # in available_turbines() weiss davon aber noch nichts -> Cache leeren.
    available_turbines.clear()
    return target


def turbine_file_name(name, rated_power_kw, rotor_diameter, hub_height):
    clean_name = _safe_turbine_name(name or "CustomTurbine")
    power_mw = rated_power_kw / 1000.0
    return (
        f"{clean_name}-{_fmt_float_for_foxes(power_mw)}MW"
        f"-D{_fmt_float_for_foxes(rotor_diameter)}"
        f"-H{_fmt_float_for_foxes(hub_height)}.csv"
    )


@dataclass(frozen=True)
class SimulationConfig:
    layout_mode: str
    n_turbines: int
    spacing_x: float
    spacing_y: float
    hub_heights: tuple[float, ...]
    plane_height: float
    grid_points: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    wind_speed: float
    wind_direction: float
    turbulence_intensity: float
    air_density: float
    atmosphere_preset: str
    vertical_profile: str
    shear_exponent: float
    roughness_length: float
    monin_obukhov_length: float
    wake_length_threshold: float
    particle_tracking: bool
    particle_release_interval_s: float
    particle_duration_s: float
    particle_dt_s: float
    particle_turbulence_scale: float
    particle_seed: int
    turbine_label: str
    wake_label: str
    # Steuert, ob die (teuren) Streamline-/Kontur-Wake-Laengen ueberhaupt
    # berechnet werden. Vorher liefen diese Berechnungen IMMER mit, auch wenn
    # sie gar nicht angezeigt wurden (z.B. im Windpark-Modus oder wenn die
    # entsprechende Checkbox deaktiviert ist). Default True, damit bestehender
    # Code ohne Angabe dieser Felder unveraendert funktioniert.
    compute_streamline_wake_length: bool = True
    compute_contour_wake_length: bool = True


@dataclass(frozen=True)
class WindClimateConfig:
    turbine_label: str
    custom_name: str
    rotor_diameter: float
    hub_height: float
    rated_power_kw: float
    cut_in_ws: float
    rated_ws: float
    cut_out_ws: float
    ct_below_rated: float
    ct_above_rated: float
    weibull_a: float
    weibull_k: float
    wind_direction: float
    direction_concentration: float
    n_speed_bins: int
    n_direction_bins: int
    turbulence_intensity: float
    air_density: float
    curve_points: tuple[tuple[float, float, float], ...] = ()


@st.cache_resource(show_spinner=False)
def _get_turbine_type(turbine_file: str):
    """Baut den FOXES-Turbinentyp aus einer Kennlinien-CSV.

    Das Parsen der CSV + Konstruktion des PCtFile-Objekts ist reine, von den
    Windzustaenden unabhaengige IO/Setup-Arbeit. Vorher wurde das bei jedem
    einzelnen run_single_state()-Aufruf neu gemacht (auch bei jedem einzelnen
    Runner-Zeitschritt). st.cache_resource haelt das FOXES-Objekt als
    Singleton im Speicher, solange sich turbine_file nicht aendert.
    """
    import foxes

    return foxes.models.turbine_types.PCtFile(turbine_file)


def run_single_state(config: SimulationConfig):
    import foxes
    import foxes.variables as FV

    mbook = foxes.ModelBook()
    turbine_file = available_turbines()[config.turbine_label]
    turbine_type = _get_turbine_type(turbine_file)
    mbook.turbine_types[turbine_type.name] = turbine_type

    farm = foxes.WindFarm()
    turbine_models = [turbine_type.name]
    reference_height = turbine_type.H

    if config.layout_mode == "Turbinenreihe":
        xy_base = np.array([0.0, 0.0])
        xy_step = np.array([config.spacing_x, config.spacing_y])
        hub_heights = _hub_heights(config, turbine_type.H)
        reference_height = float(np.mean(hub_heights))

        for i, hub_height in enumerate(hub_heights):
            farm.add_turbine(
                foxes.Turbine(
                    xy=xy_base + i * xy_step,
                    name=f"T{i}",
                    H=float(hub_height),
                    turbine_models=turbine_models,
                ),
                verbosity=0,
            )
    else:
        foxes.input.farm_layout.add_from_file(
            farm,
            "test_farm_67.csv",
            turbine_models=turbine_models,
        )

    profiles = {}
    profile_data = {}
    if config.vertical_profile == "shear":
        profiles[FV.WS] = "ShearedProfile"
        profile_data[FV.H] = reference_height
        profile_data[FV.SHEAR] = config.shear_exponent
    elif config.vertical_profile == "ABL log":
        profiles[FV.WS] = "ABLLogWsProfile"
        profile_data[FV.H] = reference_height
        profile_data[FV.Z0] = config.roughness_length
        profile_data[FV.MOL] = config.monin_obukhov_length

    states = foxes.input.states.SingleStateStates(
        ws=config.wind_speed,
        wd=config.wind_direction,
        ti=config.turbulence_intensity,
        rho=config.air_density,
        profiles=profiles,
        **profile_data,
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=WAKE_MODEL_OPTIONS[config.wake_label],
        wake_frame="rotor_wd",
        mbook=mbook,
        verbosity=0,
    )

    active_engine = foxes.get_engine(error=False)
    engine_context = nullcontext(active_engine) if active_engine is not None else foxes.Engine.new(engine_type="single")
    with engine_context:
        farm_results = algo.calc_farm()
        evaluator = foxes.output.FarmResultsEval(farm_results, algo=algo)
        evaluator.add_efficiency()

        farm_df = farm_results.to_dataframe().reset_index()
        columns = [
            FV.X,
            FV.Y,
            FV.H,
            FV.WD,
            FV.AMB_REWS,
            FV.REWS,
            FV.AMB_TI,
            FV.TI,
            FV.AMB_P,
            FV.P,
            FV.CT,
            FV.EFF,
        ]
        available_columns = [c for c in columns if c in farm_df.columns]
        results_df = farm_df[available_columns].copy()

        for col in [FV.AMB_P, FV.P]:
            if col in results_df.columns:
                results_df[col] = results_df[col] / 1000.0

        summary = {
            "power_mw": evaluator.calc_mean_farm_power() / 1000.0,
            "ambient_power_mw": evaluator.calc_mean_farm_power(ambient=True) / 1000.0,
            "efficiency_percent": evaluator.calc_farm_efficiency() * 100.0,
        }

        layout_df = _farm_layout_dataframe(farm, farm_df=farm_df, height_col=FV.H)

        flow_field = _calculate_horizontal_flow_field(
            foxes=foxes,
            FV=FV,
            algo=algo,
            farm_results=farm_results,
            plane_height=config.plane_height,
            grid_points=config.grid_points,
            x_min=config.x_min,
            x_max=config.x_max,
            y_min=config.y_min,
            y_max=config.y_max,
        )
        # Streamline-/Kontur-Wake-Laengen sind nur im Turbinenreihe-Layout
        # ueberhaupt aussagekraeftig und werden nur dort im UI angezeigt.
        # Vorher liefen die teuren Berechnungen (Streamline-Tracer,
        # Flood-Fill-Konturflaeche, Matplotlib-Konturextraktion) IMMER mit -
        # auch im Windpark-Modus und auch wenn die Anzeige-Checkbox aus war.
        compute_streamline = (
            config.layout_mode == "Turbinenreihe" and config.compute_streamline_wake_length
        )
        compute_contour = (
            config.layout_mode == "Turbinenreihe" and config.compute_contour_wake_length
        )
        wake_metrics = calculate_wake_length_metrics(
            flow_field,
            layout_df=layout_df,
            ambient_ws=config.wind_speed,
            threshold_percent=config.wake_length_threshold,
            compute_streamline=compute_streamline,
            compute_contour=compute_contour,
        )
        summary.update(wake_metrics)

        particle_paths = []
        if config.layout_mode == "Turbinenreihe" and config.particle_tracking:
            particle_paths = simulate_particle_paths(
                flow_field=flow_field,
                layout_df=layout_df,
                dt=config.particle_dt_s,
                duration=config.particle_duration_s,
                release_interval=config.particle_release_interval_s,
                turbulence_intensity=config.turbulence_intensity,
                turbulence_scale=config.particle_turbulence_scale,
                seed=config.particle_seed,
            )
            summary.update(calculate_particle_metrics(particle_paths))
        flow_field["particle_paths"] = particle_paths

    return summary, results_df, layout_df, flow_field


def run_single_turbine_yield(config: WindClimateConfig):
    import foxes
    import foxes.variables as FV

    states_df = generate_wind_climate(config)
    curve_df = create_power_ct_curve(config)

    mbook = foxes.ModelBook()
    if config.turbine_label == CUSTOM_TURBINE_LABEL:
        turbine_type = foxes.models.turbine_types.PCtFile(
            curve_df,
            name=config.custom_name or "CustomTurbine",
            D=config.rotor_diameter,
            H=config.hub_height,
            P_nominal=config.rated_power_kw,
            P_unit="kW",
        )
    else:
        turbine_type = _get_turbine_type(available_turbines()[config.turbine_label])

    mbook.turbine_types[turbine_type.name] = turbine_type

    states = foxes.input.states.StatesTable(
        data_source=states_df,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.TI: config.turbulence_intensity, FV.RHO: config.air_density},
    )

    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(
            xy=np.array([0.0, 0.0]),
            name="T0",
            H=float(turbine_type.H),
            turbine_models=[turbine_type.name],
        ),
        verbosity=0,
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Jensen_linear_k007"],
        wake_frame="rotor_wd",
        mbook=mbook,
        verbosity=0,
    )

    active_engine = foxes.get_engine(error=False)
    engine_context = nullcontext(active_engine) if active_engine is not None else foxes.Engine.new(engine_type="single")
    with engine_context:
        farm_results = algo.calc_farm()
        evaluator = foxes.output.FarmResultsEval(farm_results, algo=algo)
        mean_power_kw = evaluator.calc_mean_farm_power()
        aep_gwh = evaluator.calc_farm_yield()

    summary = {
        "mean_power_kw": mean_power_kw,
        "aep_gwh": aep_gwh,
        "capacity_factor_percent": mean_power_kw / turbine_type.P_nominal * 100.0,
        "turbine_name": turbine_type.name,
        "rotor_diameter": float(turbine_type.D),
        "hub_height": float(turbine_type.H),
        "rated_power_kw": float(turbine_type.P_nominal),
    }

    return summary, states_df, curve_df


def generate_wind_climate(config: WindClimateConfig) -> pd.DataFrame:
    ws_edges = np.linspace(0.0, config.cut_out_ws, config.n_speed_bins + 1)
    ws_centres = 0.5 * (ws_edges[:-1] + ws_edges[1:])
    speed_weights = _weibull_cdf(ws_edges[1:], config.weibull_a, config.weibull_k)
    speed_weights -= _weibull_cdf(ws_edges[:-1], config.weibull_a, config.weibull_k)
    speed_weights = np.maximum(speed_weights, 0.0)
    speed_weights = speed_weights / speed_weights.sum()

    wd_edges = np.linspace(0.0, 360.0, config.n_direction_bins + 1)
    wd_centres = 0.5 * (wd_edges[:-1] + wd_edges[1:])
    angles = np.deg2rad(wd_centres - config.wind_direction)
    direction_weights = np.exp(config.direction_concentration * np.cos(angles))
    direction_weights = direction_weights / direction_weights.sum()

    rows = []
    for wd, w_wd in zip(wd_centres, direction_weights):
        for ws, w_ws in zip(ws_centres, speed_weights):
            rows.append({"wd": wd, "ws": ws, "weight": w_wd * w_ws})

    df = pd.DataFrame(rows)
    df["weight"] = df["weight"] / df["weight"].sum()
    return df


@st.cache_data(show_spinner=False)
def create_power_ct_curve(config: WindClimateConfig) -> pd.DataFrame:
    if config.curve_points:
        curve = pd.DataFrame(config.curve_points, columns=["ws", "P", "ct"])
        curve = curve.sort_values("ws").drop_duplicates("ws", keep="last")
        return curve.reset_index(drop=True)

    if config.turbine_label != CUSTOM_TURBINE_LABEL:
        return pd.read_csv(_static_curve_path(config.turbine_label))

    ws = np.arange(0.0, max(config.cut_out_ws, 1.0) + 0.5, 0.5)
    power = np.zeros_like(ws)

    active = (ws >= config.cut_in_ws) & (ws < config.rated_ws)
    denom = max(config.rated_ws**3 - config.cut_in_ws**3, 1e-9)
    power[active] = config.rated_power_kw * (
        (ws[active] ** 3 - config.cut_in_ws**3) / denom
    )
    power[(ws >= config.rated_ws) & (ws <= config.cut_out_ws)] = config.rated_power_kw

    ct = np.zeros_like(ws)
    ct[(ws >= config.cut_in_ws) & (ws < config.rated_ws)] = config.ct_below_rated
    ct[(ws >= config.rated_ws) & (ws <= config.cut_out_ws)] = config.ct_above_rated

    return pd.DataFrame({"ws": ws, "P": power, "ct": ct})


def _weibull_cdf(ws, scale, shape):
    return 1.0 - np.exp(-((ws / scale) ** shape))


def _static_curve_path(turbine_label: str):
    return _power_ct_curve_dir() / available_turbines()[turbine_label]


def _foxes_data_dir(subdir: str) -> Path:
    """Verzeichnis der FOXES-Statikdaten - deploy-fest ueber das installierte foxes-Paket.

    Vorher wurde ein hartkodierter, repo-relativer Pfad (.../foxes/foxes/data/...) verwendet,
    der nur funktioniert, wenn das FOXES-Repo als Schwesterordner daneben liegt. Beim Hosting
    (Streamlit Cloud, JupyterHub) ist das nicht der Fall. Diese Variante findet die Daten im
    installierten foxes-Paket und faellt nur zur Not auf das alte Layout zurueck.
    """
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


def _power_ct_curve_dir() -> Path:
    return _foxes_data_dir("power_ct_curves")


def _safe_turbine_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "", name.strip())
    return safe or "CustomTurbine"


def _fmt_float_for_foxes(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "d")


def _parse_hub_height_from_file_name(file_name: str, fallback: float) -> float:
    match = re.search(r"(?:^|[-_])H([0-9]+(?:d[0-9]+)?)", Path(file_name).stem)
    if not match:
        return fallback
    return float(match.group(1).replace("d", "."))


def _hub_heights(config: SimulationConfig, default_height: float) -> np.ndarray:
    if config.hub_heights:
        heights = np.array(config.hub_heights, dtype=float)
        if len(heights) == config.n_turbines:
            return heights

    return np.full(config.n_turbines, default_height)


def _calculate_horizontal_flow_field(
    foxes,
    FV,
    algo,
    farm_results,
    plane_height: float,
    grid_points: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    flow_output = foxes.output.FlowPlots2D(algo, farm_results)
    parameters, data, _states, grid_data = flow_output.get_states_data_xy(
        FV.AMB_WS,
        data_format="numpy",
        n_img_points=(grid_points, grid_points),
        z=plane_height,
        xmin=x_min,
        xmax=x_max,
        ymin=y_min,
        ymax=y_max,
        xspace=0.0,
        yspace=0.0,
    )
    ws_index = parameters["variables"].index(FV.WS)
    wd_index = parameters["variables"].index(FV.WD)
    x_pos, y_pos, z_pos, _grid_points = grid_data
    ws = data[0, :, :, ws_index]

    flow_field = {
        "x": x_pos,
        "y": y_pos,
        "z": float(z_pos),
        "ws": ws,
        "wd": data[0, :, :, wd_index],
    }
    if FV.AMB_WS in parameters["variables"]:
        flow_field["amb_ws"] = data[0, :, :, parameters["variables"].index(FV.AMB_WS)]
    else:
        flow_field["amb_ws"] = np.full_like(ws, np.nan)

    return flow_field


def _wake_deficit_percent(ws, ambient_ws):
    ws_arr, ambient_arr = np.broadcast_arrays(
        np.asarray(ws, dtype=float),
        np.asarray(ambient_ws, dtype=float),
    )
    deficit = np.zeros_like(ws_arr, dtype=float)
    valid = np.isfinite(ambient_arr) & (ambient_arr > 1e-9)
    np.divide(ambient_arr - ws_arr, ambient_arr, out=deficit, where=valid)
    return np.maximum(0.0, deficit * 100.0)


def calculate_wake_length_metrics(
    flow_field,
    layout_df: pd.DataFrame,
    ambient_ws: float,
    threshold_percent: float,
    compute_streamline: bool = True,
    compute_contour: bool = True,
):
    x_values = flow_field["x"]
    y_values = flow_field["y"]
    ws = flow_field["ws"]
    if len(x_values) == 0 or len(y_values) == 0:
        return {
            "wake_length_m": 0.0,
            "wake_streamline_length_m": 0.0,
            "wake_contour_max_length_m": 0.0,
            "wake_contour_area_m2": 0.0,
            "wake_contour_count": 0,
            "wake_min_ws": np.nan,
            "wake_max_deficit_percent": np.nan,
        }

    centre_y_index = int(np.argmin(np.abs(y_values)))
    downstream = x_values >= 0.0
    line_x = x_values[downstream]
    line_ws = ws[downstream, centre_y_index]
    if "amb_ws" in flow_field and not np.all(np.isnan(flow_field["amb_ws"])):
        line_ambient_ws = flow_field["amb_ws"][downstream, centre_y_index]
    else:
        line_ambient_ws = np.full_like(line_ws, ambient_ws)
    deficit_percent = _wake_deficit_percent(line_ws, line_ambient_ws)
    active = deficit_percent >= threshold_percent

    if not np.any(active):
        wake_length = 0.0
    else:
        wake_length = float(line_x[np.where(active)[0].max()])
    flow_field["wake_midline"] = {"x": [0.0, wake_length], "y": [0.0, 0.0]}

    # Streamline-Tracer und Kontur-/Flaechenberechnung sind mit Abstand die
    # teuersten Teile hier (Python-Schleifen pro Turbine bzw. pro Gitterzelle
    # + Matplotlib/contourpy-Aufruf). Wenn der Aufrufer sie nicht braucht
    # (Windpark-Modus oder Anzeige deaktiviert), werden sie komplett
    # uebersprungen statt nur das Ergebnis zu verwerfen.
    if compute_streamline:
        streamline_metrics = _calculate_streamline_wake_length(
            flow_field,
            layout_df=layout_df,
            ambient_ws=ambient_ws,
            threshold_percent=threshold_percent,
        )
    else:
        streamline_metrics = {"wake_streamline_length_m": 0.0}

    if compute_contour:
        contour_metrics = _calculate_contour_wake_length(
            flow_field,
            ambient_ws=ambient_ws,
            threshold_percent=threshold_percent,
        )
    else:
        contour_metrics = {
            "wake_contour_max_length_m": 0.0,
            "wake_contour_area_m2": 0.0,
            "wake_contour_count": 0,
        }

    return {
        "wake_length_m": wake_length,
        **streamline_metrics,
        **contour_metrics,
        "wake_min_ws": float(np.nanmin(line_ws)),
        "wake_max_deficit_percent": float(np.nanmax(deficit_percent)),
    }


def _calculate_streamline_wake_length(flow_field, layout_df: pd.DataFrame, ambient_ws: float, threshold_percent: float):
    x_values = flow_field["x"]
    y_values = flow_field["y"]
    ws_grid = flow_field["ws"]
    wd_grid = flow_field["wd"]
    amb_ws_grid = flow_field.get("amb_ws")
    step = max(float(np.nanmedian(np.diff(x_values))), 1.0)

    max_steps = int(max((x_values[-1] - x_values[0]) / step, 1)) + 2
    wake_paths = []
    t0_length = 0.0

    for ti, turbine in layout_df.iterrows():
        x = float(turbine["x"])
        y = float(turbine["y"])
        distance = 0.0
        last_active_distance = 0.0
        path_x = [x]
        path_y = [y]

        for _ in range(max_steps):
            local_ws = _interp_grid(x_values, y_values, ws_grid, x, y)
            local_wd = _interp_grid(x_values, y_values, wd_grid, x, y)
            if amb_ws_grid is not None and not np.all(np.isnan(amb_ws_grid)):
                local_ambient_ws = _interp_grid(x_values, y_values, amb_ws_grid, x, y)
            else:
                local_ambient_ws = ambient_ws
            if np.isnan(local_ws) or np.isnan(local_wd) or np.isnan(local_ambient_ws):
                break

            deficit_percent = float(_wake_deficit_percent(local_ws, local_ambient_ws))
            if deficit_percent >= threshold_percent:
                last_active_distance = distance

            wd_rad = np.deg2rad(local_wd)
            x += -np.sin(wd_rad) * step
            y += -np.cos(wd_rad) * step
            distance += step

            if x < x_values[0] or x > x_values[-1] or y < y_values[0] or y > y_values[-1]:
                break

            path_x.append(x)
            path_y.append(y)

        if last_active_distance > 0.0:
            n_active = min(int(np.floor(last_active_distance / step)) + 1, len(path_x))
            wake_path = {
                "turbine": str(turbine["Turbine"]),
                "x": path_x[:n_active],
                "y": path_y[:n_active],
            }
        else:
            wake_path = {"turbine": str(turbine["Turbine"]), "x": [], "y": []}

        wake_paths.append(wake_path)
        if ti == 0:
            t0_length = last_active_distance

    flow_field["all_turbine_streamline_wakes"] = wake_paths
    flow_field["first_turbine_streamline_wake"] = wake_paths[0] if wake_paths else {"x": [], "y": []}
    return {"wake_streamline_length_m": float(t0_length)}


def _calculate_contour_wake_length(flow_field, ambient_ws: float, threshold_percent: float):
    x_values = flow_field["x"]
    y_values = flow_field["y"]
    if "amb_ws" in flow_field and not np.all(np.isnan(flow_field["amb_ws"])):
        ambient_reference = flow_field["amb_ws"]
    else:
        ambient_reference = np.full_like(flow_field["ws"], ambient_ws)
    deficit = _wake_deficit_percent(flow_field["ws"], ambient_reference)
    active = (deficit >= threshold_percent) & (x_values[:, None] >= 0.0)
    flow_field["wake_deficit_percent"] = deficit
    flow_field["wake_deficit_contours"] = _extract_deficit_contours(
        x_values,
        y_values,
        deficit,
        threshold_percent,
    )

    dx = float(np.nanmedian(np.diff(x_values))) if len(x_values) > 1 else 0.0
    dy = float(np.nanmedian(np.diff(y_values))) if len(y_values) > 1 else 0.0
    cell_area = abs(dx * dy)

    max_length = 0.0
    max_area = 0.0
    count = 0

    if np.any(active):
        # Vektorisierte Connected-Component-Suche (scipy, C-implementiert)
        # statt manuellem Stack-Flood-Fill in reinem Python. Vorher skalierte
        # das mit O(Gitterzellen) Python-Overhead (bis zu 150x150 = 22500
        # Zellen); jetzt ist die teure Arbeit vektorisiert und der
        # verbleibende Python-Loop laeuft nur ueber die Anzahl gefundener
        # Wake-Regionen (typischerweise wenige).
        labeled, count = ndimage.label(active)
        label_ids = np.arange(1, count + 1)
        cell_counts = ndimage.sum(active, labeled, index=label_ids)
        y_grid = np.broadcast_to(y_values[None, :], active.shape)
        y_means = ndimage.mean(y_grid, labeled, index=label_ids)
        bboxes = ndimage.find_objects(labeled)

        for _label_id, cells, y_mid, bbox in zip(label_ids, cell_counts, y_means, bboxes):
            if bbox is None:
                continue
            ix_slice, _iy_slice = bbox
            x_lo = float(x_values[ix_slice.start])
            x_hi = float(x_values[ix_slice.stop - 1])
            length = x_hi - max(x_lo, 0.0)
            area = float(cells) * cell_area
            if length > max_length:
                max_length = length
                max_area = area
                flow_field["wake_contour_max_line"] = {
                    "x": [float(max(x_lo, 0.0)), x_hi],
                    "y": [float(y_mid), float(y_mid)],
                }

    return {
        "wake_contour_max_length_m": max_length,
        "wake_contour_area_m2": max_area,
        "wake_contour_count": int(count),
    }


def _extract_deficit_contours(x_values, y_values, deficit_percent, threshold_percent):
    """Isolinien fuer den Wake-Defizit-Schwellwert.

    Nutzt contourpy direkt (die Bibliothek, die matplotlib intern fuer
    ax.contour() aufruft) statt bei jedem Simulationslauf eine komplette
    Matplotlib-Figure/Axes samt Agg-Backend aufzubauen und sofort wieder zu
    verwerfen, nur um an die Isolinien-Koordinaten zu kommen. Liefert
    identische Koordinaten, ist aber in Benchmarks ca. 20-30x schneller.
    """
    try:
        generator = contourpy.contour_generator(
            x=x_values,
            y=y_values,
            z=deficit_percent.T,
            name="serial",
        )
        lines = generator.lines(float(threshold_percent))
    except Exception:
        return []

    contours = []
    for segment in lines:
        segment = np.asarray(segment, dtype=float)
        if segment.shape[0] >= 2:
            contours.append(
                {
                    "x": segment[:, 0].tolist(),
                    "y": segment[:, 1].tolist(),
                }
            )
    return contours


def simulate_particle_paths(
    flow_field,
    layout_df: pd.DataFrame,
    dt: float,
    duration: float,
    release_interval: float,
    turbulence_intensity: float,
    turbulence_scale: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    x_values = flow_field["x"]
    y_values = flow_field["y"]
    ws_grid = flow_field["ws"]
    wd_grid = flow_field["wd"]

    if dt <= 0.0 or duration <= 0.0 or release_interval <= 0.0:
        return []

    release_times = np.arange(0.0, duration + 0.5 * release_interval, release_interval)
    steps = int(np.ceil(duration / dt))
    paths = []

    for _, turbine in layout_df.iterrows():
        for release_time in release_times:
            x = float(turbine["x"])
            y = float(turbine["y"])
            z = float(turbine["Nabenhoehe"])
            xs = [x]
            ys = [y]
            zs = [z]

            start_step = int(np.round(release_time / dt))
            for _ in range(start_step, steps):
                local_ws = _interp_grid(x_values, y_values, ws_grid, x, y)
                local_wd = _interp_grid(x_values, y_values, wd_grid, x, y)
                if np.isnan(local_ws) or np.isnan(local_wd):
                    break

                wd_rad = np.deg2rad(local_wd)
                ux = -np.sin(wd_rad)
                uy = -np.cos(wd_rad)
                cross_x = -uy
                cross_y = ux
                sigma = turbulence_scale * turbulence_intensity * local_ws

                stream_fluct = rng.normal(0.0, 0.25 * sigma)
                cross_fluct = rng.normal(0.0, sigma)
                vertical_fluct = rng.normal(0.0, 0.55 * sigma)

                x += ((local_ws + stream_fluct) * ux + cross_fluct * cross_x) * dt
                y += ((local_ws + stream_fluct) * uy + cross_fluct * cross_y) * dt
                z = max(0.0, z + vertical_fluct * dt)

                if (
                    x < x_values[0]
                    or x > x_values[-1]
                    or y < y_values[0]
                    or y > y_values[-1]
                ):
                    break

                xs.append(x)
                ys.append(y)
                zs.append(z)

            if len(xs) > 1:
                paths.append(
                    {
                        "turbine": str(turbine["Turbine"]),
                        "release_time": float(release_time),
                        "x": xs,
                        "y": ys,
                        "z": zs,
                    }
                )

    return paths


def calculate_particle_metrics(particle_paths):
    if not particle_paths:
        return {
            "particle_path_count": 0,
            "particle_lateral_spread_m": 0.0,
            "particle_vertical_spread_m": 0.0,
        }

    y_end = np.array([path["y"][-1] for path in particle_paths], dtype=float)
    z_end = np.array([path["z"][-1] for path in particle_paths], dtype=float)
    return {
        "particle_path_count": int(len(particle_paths)),
        "particle_lateral_spread_m": float(np.nanpercentile(y_end, 95) - np.nanpercentile(y_end, 5)),
        "particle_vertical_spread_m": float(np.nanpercentile(z_end, 95) - np.nanpercentile(z_end, 5)),
    }


def _interp_grid(x_values, y_values, values, x, y):
    if x < x_values[0] or x > x_values[-1] or y < y_values[0] or y > y_values[-1]:
        return np.nan

    ix = np.searchsorted(x_values, x) - 1
    iy = np.searchsorted(y_values, y) - 1
    ix = int(np.clip(ix, 0, len(x_values) - 2))
    iy = int(np.clip(iy, 0, len(y_values) - 2))

    x0, x1 = x_values[ix], x_values[ix + 1]
    y0, y1 = y_values[iy], y_values[iy + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    v00 = values[ix, iy]
    v10 = values[ix + 1, iy]
    v01 = values[ix, iy + 1]
    v11 = values[ix + 1, iy + 1]
    return float(
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def _farm_layout_dataframe(farm, farm_df=None, height_col=None) -> pd.DataFrame:
    rows = []
    for i, turbine in enumerate(farm.turbines):
        hub_height = getattr(turbine, "H", None)
        if hub_height is None and farm_df is not None and height_col in farm_df.columns:
            hub_height = farm_df[height_col].iloc[i]

        rows.append(
            {
                "Turbine": getattr(turbine, "name", f"T{i}"),
                "x": float(turbine.xy[0]),
                "y": float(turbine.xy[1]),
                "Nabenhoehe": float(hub_height) if hub_height is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)
