"""Hauptbereich fuer den Arbeitsbereich 'Turbine & Windklima'."""

import streamlit as st

from foxes_runner import (
    CUSTOM_TURBINE_LABEL,
    WindClimateConfig,
    create_power_ct_curve,
    run_single_turbine_yield,
    save_custom_turbine_curve,
    turbine_file_name,
)
from figures import _power_curve_figure, _wind_rose_figure, _wind_speed_distribution_figure
from state import ensure_result


def render_climate_tab(values):
    if not (values["cut_in_ws"] < values["rated_ws"] < values["cut_out_ws"]):
        st.error("Bitte waehle Einschaltwind < Nennwind < Abschaltwind.")
        st.stop()

    base_climate_config = WindClimateConfig(
        turbine_label=values["climate_turbine_label"],
        custom_name=values["custom_name"],
        rotor_diameter=values["rotor_diameter"],
        hub_height=values["custom_hub_height"],
        rated_power_kw=values["rated_power_kw"],
        cut_in_ws=values["cut_in_ws"],
        rated_ws=values["rated_ws"],
        cut_out_ws=values["cut_out_ws"],
        ct_below_rated=values["ct_below_rated"],
        ct_above_rated=values["ct_above_rated"],
        weibull_a=values["weibull_a"],
        weibull_k=values["weibull_k"],
        wind_direction=values["climate_wind_direction"],
        direction_concentration=values["direction_concentration"],
        n_speed_bins=values["n_speed_bins"],
        n_direction_bins=values["n_direction_bins"],
        turbulence_intensity=values["climate_ti"],
        air_density=values["climate_rho"],
    )
    # create_power_ct_curve ist jetzt st.cache_data-gecacht (siehe
    # foxes_runner.py) - bei unveraenderten Reglern liefert dieser Aufruf
    # sofort das gecachte Ergebnis statt CSV/Weibull-Berechnung erneut
    # auszufuehren.
    default_curve_df = create_power_ct_curve(base_climate_config)

    curve_signature = (
        values["climate_turbine_label"],
        values["custom_name"],
        values["rotor_diameter"],
        values["custom_hub_height"],
        values["rated_power_kw"],
        values["cut_in_ws"],
        values["rated_ws"],
        values["cut_out_ws"],
        values["ct_below_rated"],
        values["ct_above_rated"],
    )
    if (
        "curve_table" not in st.session_state
        or st.session_state.get("curve_signature") != curve_signature
    ):
        st.session_state.curve_signature = curve_signature
        st.session_state.curve_table = default_curve_df.copy()

    curve_table = st.session_state.curve_table.copy()
    curve_points = ()
    if values["climate_turbine_label"] == CUSTOM_TURBINE_LABEL:
        curve_points = tuple(
            tuple(row)
            for row in curve_table[["ws", "P", "ct"]]
            .astype(float)
            .to_numpy()
            .tolist()
        )

    climate_config_kwargs = dict(base_climate_config.__dict__)
    climate_config_kwargs["curve_points"] = curve_points
    climate_config = WindClimateConfig(**climate_config_kwargs)

    climate_summary, states_df, curve_df = ensure_result(
        config=climate_config,
        config_key="last_climate_config",
        result_key="climate_result",
        error_key="climate_error",
        compute_fn=run_single_turbine_yield,
        run_clicked=values["run_climate_clicked"],
        spinner_text="FOXES rechnet Einzelturbinen-AEP...",
        waiting_text="Starte eine Windklima-Analyse.",
    )

    st.subheader("Turbine")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Turbine", climate_summary["turbine_name"])
    metric_cols[1].metric("Nennleistung", f"{climate_summary['rated_power_kw']:.0f} kW")
    metric_cols[2].metric("Rotordurchmesser", f"{climate_summary['rotor_diameter']:.1f} m")
    metric_cols[3].metric("Nabenhoehe", f"{climate_summary['hub_height']:.1f} m")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Mittlere Leistung", f"{climate_summary['mean_power_kw']:.1f} kW")
    metric_cols[1].metric("Jahresertrag", f"{climate_summary['aep_gwh']:.3f} GWh")
    metric_cols[2].metric("Kapazitaetsfaktor", f"{climate_summary['capacity_factor_percent']:.1f} %")

    left, right = st.columns([1.15, 1.0])
    with left:
        st.subheader("Windrose")
        st.plotly_chart(_wind_rose_figure(states_df), use_container_width=True)
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.subheader("Windgeschwindigkeits-Haeufigkeit")
        st.plotly_chart(
            _wind_speed_distribution_figure(
                states_df,
                st.session_state.last_climate_config.weibull_a,
                st.session_state.last_climate_config.weibull_k,
            ),
            use_container_width=True,
        )
    with right:
        st.subheader("Leistungs- und ct-Kennlinie")
        st.plotly_chart(_power_curve_figure(curve_df), use_container_width=True)
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.subheader("Kennlinien-Stuetzpunkte")
        if values["climate_turbine_label"] == CUSTOM_TURBINE_LABEL:
            expected_name = turbine_file_name(
                climate_config.custom_name,
                climate_config.rated_power_kw,
                climate_config.rotor_diameter,
                climate_config.hub_height,
            )
            st.caption(f"FOXES-Dateiname: `{expected_name}`")
            edited_curve = st.data_editor(
                st.session_state.curve_table,
                hide_index=True,
                use_container_width=True,
                height=300,
                column_config={
                    "ws": st.column_config.NumberColumn(
                        "     ws",
                        min_value=0.0,
                        max_value=60.0,
                        step=0.5,
                        format="%.2f",
                        width="small",
                    ),
                    "P": st.column_config.NumberColumn(
                        "      P",
                        min_value=0.0,
                        step=10.0,
                        format="%.1f",
                        width="small",
                    ),
                    "ct": st.column_config.NumberColumn(
                        "      ct",
                        min_value=0.0,
                        max_value=1.5,
                        step=0.01,
                        format="%.3f",
                        width="small",
                    ),
                },
                num_rows="dynamic",
            )
            st.session_state.curve_table = (
                edited_curve[["ws", "P", "ct"]]
                .dropna()
                .astype(float)
                .sort_values("ws")
                .reset_index(drop=True)
            )
            st.markdown("**Turbine dauerhaft speichern**")
            if st.button(
                "Als FOXES-Turbine speichern",
                type="primary",
                use_container_width=True,
            ):
                saved_path = save_custom_turbine_curve(
                    climate_config,
                    st.session_state.curve_table,
                )
                st.success(f"Gespeichert: {saved_path.name}")
                st.info("Nach dem naechsten Neuladen der App erscheint die Turbine in der Auswahlliste.")
        else:
            st.dataframe(curve_df, use_container_width=True, hide_index=True, height=300)

    st.stop()
