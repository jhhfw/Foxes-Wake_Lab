"""Alle Plotly-Figure-Builder der App an einem Ort.

Reine Funktionen (Daten -> plotly.graph_objects.Figure), ohne Streamlit-
Abhaengigkeit. Vorher standen diese Funktionen verstreut in der 2000+
Zeilen app.py; hier lassen sie sich unabhaengig lesen, testen und anpassen.

Performance-/Darstellungs-Aenderungen gegenueber der urspruenglichen Version:
- Traces mit potenziell vielen Punkten/Segmenten (Partikelbahnen,
  Windrichtungspfeile, Wake-Paketbahnen, Streamline-Wake-Pfade) nutzen jetzt
  go.Scattergl (WebGL) statt go.Scatter (SVG) fuer fluessigeres Pan/Zoom,
  besonders relevant im animierten Runner, wo die Figure bei jedem Tick neu
  aufgebaut wird.
- Neue _wake_field_comparison_figure() fuer den Dynamic-Runner-Vergleich
  (dynamisches Feld vs. FOXES-Referenz) mit gemeinsamen/verknuepften Achsen,
  damit Zoom/Pan in einem Panel synchron im anderen mitlaeuft.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _build_wind_arrow_lines(
    x_values,
    y_values,
    wind_directions,
    spacing,
    length,
    head_length,
    head_angle_deg,
):
    x_targets = np.arange(x_values[0], x_values[-1] + spacing, spacing)
    y_targets = np.arange(y_values[0], y_values[-1] + spacing, spacing)
    x_indices = np.unique([np.abs(x_values - x).argmin() for x in x_targets])
    y_indices = np.unique([np.abs(y_values - y).argmin() for y in y_targets])

    line_x = []
    line_y = []
    head_angle = np.deg2rad(head_angle_deg)

    for ix in x_indices:
        for iy in y_indices:
            x0 = float(x_values[ix])
            y0 = float(y_values[iy])
            wd_rad = np.deg2rad(float(wind_directions[ix, iy]))

            # FOXES wind direction is "from"; the vector below points downstream.
            u = -np.sin(wd_rad)
            v = -np.cos(wd_rad)
            x1 = x0 + length * u
            y1 = y0 + length * v

            left_u = np.cos(head_angle) * (-u) - np.sin(head_angle) * (-v)
            left_v = np.sin(head_angle) * (-u) + np.cos(head_angle) * (-v)
            right_u = np.cos(-head_angle) * (-u) - np.sin(-head_angle) * (-v)
            right_v = np.sin(-head_angle) * (-u) + np.cos(-head_angle) * (-v)

            xl = x1 + head_length * left_u
            yl = y1 + head_length * left_v
            xr = x1 + head_length * right_u
            yr = y1 + head_length * right_v

            line_x.extend([x0, x1, None, x1, xl, None, x1, xr, None])
            line_y.extend([y0, y1, None, y1, yl, None, y1, yr, None])

    return line_x, line_y


def _add_wake_field_traces(
    fig,
    flow_field,
    layout_df,
    row=None,
    col=None,
    show_wind_arrows=True,
    show_midline_wake_length=False,
    show_streamline_wake_length=False,
    show_contour_wake_length=False,
    show_particles=True,
    heatmap_scale_mode="Automatisch",
    heatmap_ws_max=25.0,
    showscale=True,
):
    """Fuegt alle Wake-Feld-Traces (Heatmap, Pfeile, Wake-Laengen, Partikel,
    Turbinen) einer bestehenden Figure hinzu. row/col adressieren ein
    Subplot (fuer die Vergleichsansicht); bei row=col=None wird auf eine
    normale (nicht per make_subplots erzeugte) Figure gezeichnet.
    """
    heatmap_kwargs = {}
    if heatmap_scale_mode == "Manuell":
        heatmap_kwargs = {"zmin": 0.0, "zmax": float(heatmap_ws_max)}

    fig.add_trace(
        go.Heatmap(
            x=flow_field["x"],
            y=flow_field["y"],
            z=flow_field["ws"].T,
            colorscale="Viridis",
            colorbar={"title": "WS [m/s]"},
            showscale=showscale,
            hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<br>WS=%{z:.2f} m/s<extra></extra>",
            **heatmap_kwargs,
        ),
        row=row,
        col=col,
    )
    if show_wind_arrows:
        arrow_x, arrow_y = _build_wind_arrow_lines(
            x_values=flow_field["x"],
            y_values=flow_field["y"],
            wind_directions=flow_field["wd"],
            spacing=200.0,
            length=70.0,
            head_length=22.0,
            head_angle_deg=28.0,
        )
        # Viele kurze Liniensegmente -> Scattergl (WebGL) statt Scatter (SVG)
        # fuer deutlich fluessigeres Rendering/Pan/Zoom im Browser.
        fig.add_trace(
            go.Scattergl(
                x=arrow_x,
                y=arrow_y,
                mode="lines",
                line={"color": "rgba(0,0,0,0.62)", "width": 1},
                hoverinfo="skip",
                showlegend=False,
                name="Windrichtung",
            ),
            row=row,
            col=col,
        )

    gust_fronts = flow_field.get("gust_fronts", [])
    if gust_fronts:
        centre_x, centre_y, edge_x, edge_y = [], [], [], []
        for front in gust_fronts:
            if front.get("role") == "centre":
                centre_x.extend(front["x"] + [None])
                centre_y.extend(front["y"] + [None])
            else:
                edge_x.extend(front["x"] + [None])
                edge_y.extend(front["y"] + [None])
        if edge_x:
            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line={"color": "rgba(255,255,255,0.85)", "width": 2, "dash": "dot"},
                    hoverinfo="skip",
                    showlegend=False,
                    name="Boeenband",
                ),
                row=row,
                col=col,
            )
        if centre_x:
            fig.add_trace(
                go.Scatter(
                    x=centre_x,
                    y=centre_y,
                    mode="lines",
                    line={"color": "rgba(230,0,120,0.95)", "width": 4, "dash": "dash"},
                    hovertemplate="Boeenfront<extra></extra>",
                    showlegend=False,
                    name="Boeenfront",
                ),
                row=row,
                col=col,
            )

    midline = flow_field.get("wake_midline")
    if show_midline_wake_length and midline:
        fig.add_trace(
            go.Scatter(
                x=midline["x"],
                y=midline["y"],
                mode="lines",
                line={"color": "rgba(0, 89, 255, 0.95)", "width": 4, "dash": "dash"},
                hovertemplate="Mittellinien-Wake-Laenge<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
                showlegend=False,
                name="Wake-Laenge Mittellinie",
            ),
            row=row,
            col=col,
        )

    all_streamlines = flow_field.get("all_turbine_streamline_wakes")
    if show_streamline_wake_length and all_streamlines:
        streamline_x, streamline_y = [], []
        for streamline in all_streamlines:
            if streamline["x"]:
                streamline_x.extend(streamline["x"] + [None])
                streamline_y.extend(streamline["y"] + [None])
        fig.add_trace(
            go.Scattergl(
                x=streamline_x,
                y=streamline_y,
                mode="lines",
                line={"color": "rgba(255,255,255,0.95)", "width": 3, "dash": "dash"},
                hovertemplate="Streamline-Defizitpfad<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
                showlegend=False,
                name="Streamline-Defizitpfade",
            ),
            row=row,
            col=col,
        )

    contour_line = flow_field.get("wake_contour_max_line")
    if show_contour_wake_length and flow_field.get("wake_deficit_contours"):
        contour_x, contour_y = [], []
        for contour in flow_field["wake_deficit_contours"]:
            contour_x.extend(contour["x"] + [None])
            contour_y.extend(contour["y"] + [None])
        fig.add_trace(
            go.Scatter(
                x=contour_x,
                y=contour_y,
                mode="lines",
                line={"color": "rgba(220, 0, 0, 0.95)", "width": 2, "dash": "dash"},
                hoverinfo="skip",
                showlegend=False,
                name="Defizit-Isolinie",
            ),
            row=row,
            col=col,
        )
    if show_contour_wake_length and contour_line:
        fig.add_trace(
            go.Scatter(
                x=contour_line["x"],
                y=contour_line["y"],
                mode="lines",
                line={"color": "rgba(220, 0, 0, 0.95)", "width": 4, "dash": "dash"},
                hovertemplate="Max. Kontur-Laenge<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
                showlegend=False,
                name="Max. Kontur-Laenge",
            ),
            row=row,
            col=col,
        )

    particle_paths = flow_field.get("particle_paths", [])
    if show_particles and particle_paths:
        path_x, path_y = [], []
        for path in particle_paths:
            path_x.extend(path["x"] + [None])
            path_y.extend(path["y"] + [None])
        # Bis zu Hunderte Partikel-/Paketbahnen mit vielen Punkten ->
        # Scattergl statt Scatter.
        fig.add_trace(
            go.Scattergl(
                x=path_x,
                y=path_y,
                mode="lines",
                line={"color": "rgba(15, 15, 15, 0.38)", "width": 1},
                hoverinfo="skip",
                showlegend=False,
                name="Partikelbahnen",
            ),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            x=layout_df["x"],
            y=layout_df["y"],
            mode="markers+text",
            text=layout_df["Turbine"],
            textposition="top center",
            marker={
                "size": 11,
                "color": "white",
                "line": {"color": "black", "width": 1.5},
            },
            customdata=layout_df[["Nabenhoehe"]].to_numpy(),
            hovertemplate="Turbine=%{text}<br>x=%{x:.1f} m<br>y=%{y:.1f} m<br>H=%{customdata[0]:.1f} m<extra></extra>",
            name="Turbinen",
        ),
        row=row,
        col=col,
    )


def _wake_field_figure(
    flow_field,
    layout_df,
    show_wind_arrows=True,
    show_midline_wake_length=False,
    show_streamline_wake_length=False,
    show_contour_wake_length=False,
    show_particles=True,
    heatmap_scale_mode="Automatisch",
    heatmap_ws_max=25.0,
    height=680,
):
    fig = go.Figure()
    _add_wake_field_traces(
        fig,
        flow_field,
        layout_df,
        show_wind_arrows=show_wind_arrows,
        show_midline_wake_length=show_midline_wake_length,
        show_streamline_wake_length=show_streamline_wake_length,
        show_contour_wake_length=show_contour_wake_length,
        show_particles=show_particles,
        heatmap_scale_mode=heatmap_scale_mode,
        heatmap_ws_max=heatmap_ws_max,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        title="y [m]",
        range=[float(flow_field["y"][0]), float(flow_field["y"][-1])],
    )
    fig.update_xaxes(title="x [m]", range=[float(flow_field["x"][0]), float(flow_field["x"][-1])])
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        showlegend=False,
    )
    return fig


def _wake_field_comparison_figure(
    left_flow_field,
    left_layout_df,
    left_title,
    right_flow_field,
    right_layout_df,
    right_title,
    show_wind_arrows=True,
    show_particles=True,
    heatmap_scale_mode="Automatisch",
    heatmap_ws_max=25.0,
    height=560,
):
    """Dynamisches Feld und stationaere FOXES-Referenz nebeneinander mit
    gemeinsamen, verknuepften Achsen (Zoom/Pan in einem Panel wirkt sich auf
    das andere aus) -> direkter visueller Vergleich statt zwei unabhaengiger
    Plots.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=(left_title, right_title),
    )
    _add_wake_field_traces(
        fig,
        left_flow_field,
        left_layout_df,
        row=1,
        col=1,
        show_wind_arrows=show_wind_arrows,
        show_particles=show_particles,
        heatmap_scale_mode=heatmap_scale_mode,
        heatmap_ws_max=heatmap_ws_max,
        showscale=True,
    )
    _add_wake_field_traces(
        fig,
        right_flow_field,
        right_layout_df,
        row=1,
        col=2,
        show_wind_arrows=show_wind_arrows,
        show_particles=False,
        heatmap_scale_mode=heatmap_scale_mode,
        heatmap_ws_max=heatmap_ws_max,
        showscale=False,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="y [m]", row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_xaxes(title="x [m]", row=1, col=1)
    fig.update_xaxes(title="x [m]", row=1, col=2)
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        showlegend=False,
    )
    return fig


def _wind_rose_figure(states_df):
    rose_df = states_df.copy()
    rose_df["Haeufigkeit [%]"] = rose_df["weight"] * 100.0
    rose_df["Windrichtung [deg]"] = rose_df["wd"]
    rose_df["Windgeschwindigkeit [m/s]"] = rose_df["ws"].round(1)

    fig = go.Figure()
    speed_classes = sorted(rose_df["Windgeschwindigkeit [m/s]"].unique())
    colors = [
        "#fff7bc",
        "#fee391",
        "#fec44f",
        "#fe9929",
        "#ec7014",
        "#cc4c02",
        "#993404",
        "#662506",
    ]
    color_values = np.interp(
        np.arange(len(speed_classes)),
        [0, max(len(speed_classes) - 1, 1)],
        [0, len(colors) - 1],
    )
    speed_colors = [colors[int(round(v))] for v in color_values]

    for ws, color in zip(speed_classes, speed_colors):
        part = rose_df[rose_df["Windgeschwindigkeit [m/s]"] == ws]
        fig.add_trace(
            go.Barpolar(
                r=part["Haeufigkeit [%]"],
                theta=part["Windrichtung [deg]"],
                name=f"{ws:g} m/s",
                marker_color=color,
                hovertemplate="WD=%{theta:.0f} deg<br>Haeufigkeit=%{r:.2f} %<extra></extra>",
            )
        )
    fig.update_layout(
        height=420,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        polar={
            "angularaxis": {"direction": "clockwise", "rotation": 90},
            "radialaxis": {"ticksuffix": " %"},
        },
        legend={"title": {"text": "WS"}},
    )
    return fig


def _wind_speed_distribution_figure(states_df, weibull_a, weibull_k):
    speed_df = (
        states_df.groupby("ws", as_index=False)["weight"]
        .sum()
        .sort_values("ws")
        .reset_index(drop=True)
    )
    speed_df["Haeufigkeit [%]"] = speed_df["weight"] * 100.0

    ws_values = speed_df["ws"].to_numpy()
    if len(ws_values) > 1:
        bin_width = float(np.median(np.diff(ws_values)))
    else:
        bin_width = 1.0

    ws_line = np.linspace(0.0, max(ws_values.max() + bin_width, 1.0), 300)
    pdf_percent_per_bin = (
        (weibull_k / weibull_a)
        * (ws_line / weibull_a) ** (weibull_k - 1.0)
        * np.exp(-((ws_line / weibull_a) ** weibull_k))
        * bin_width
        * 100.0
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=speed_df["ws"],
            y=speed_df["Haeufigkeit [%]"],
            width=bin_width * 0.85,
            marker_color="#7aa6c2",
            name="Klassenhaeufigkeit",
            hovertemplate="WS=%{x:.2f} m/s<br>Haeufigkeit=%{y:.2f} %<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ws_line,
            y=pdf_percent_per_bin,
            mode="lines",
            line={"color": "#222", "width": 2},
            name="Weibull-Verteilungsfunktion",
            hovertemplate="WS=%{x:.2f} m/s<br>Haeufigkeit=%{y:.2f} % pro Bin<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 0, "r": 0, "t": 15, "b": 0},
        xaxis={"title": "Windgeschwindigkeit [m/s]"},
        yaxis={"title": "Haeufigkeit [% pro Bin]"},
        legend={"orientation": "h"},
    )
    return fig


def _power_curve_figure(curve_df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve_df["ws"],
            y=curve_df["P"],
            mode="lines",
            name="Leistung",
            line={"color": "#1f77b4", "width": 3},
            hovertemplate="WS=%{x:.1f} m/s<br>P=%{y:.1f} kW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve_df["ws"],
            y=curve_df["ct"],
            mode="lines",
            name="ct",
            yaxis="y2",
            line={"color": "#444", "width": 2},
            hovertemplate="WS=%{x:.1f} m/s<br>ct=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 15, "b": 0},
        xaxis={"title": "Windgeschwindigkeit [m/s]"},
        yaxis={"title": "Leistung [kW]"},
        yaxis2={"title": "ct [-]", "overlaying": "y", "side": "right", "range": [0, 1.1]},
        legend={"orientation": "h"},
    )
    return fig


def _runner_power_figure(history_df):
    fig = go.Figure()
    if not history_df.empty:
        fig.add_trace(
            go.Scatter(
                x=history_df["time_h"],
                y=history_df["power_mw"],
                mode="lines+markers",
                name="Parkleistung",
                line={"color": "#1f77b4", "width": 2},
                hovertemplate="t=%{x:.2f} h<br>P=%{y:.2f} MW<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=history_df["time_h"],
                y=history_df["ambient_power_mw"],
                mode="lines",
                name="ohne Wake",
                line={"color": "#777", "width": 1.5, "dash": "dash"},
                hovertemplate="t=%{x:.2f} h<br>P0=%{y:.2f} MW<extra></extra>",
            )
        )
    fig.update_layout(
        height=330,
        margin={"l": 0, "r": 0, "t": 15, "b": 0},
        xaxis={"title": "simulierte Zeit [h]"},
        yaxis={"title": "Leistung [MW]"},
        legend={"orientation": "h"},
    )
    return fig


def _runner_wake_loss_figure(history_df):
    fig = go.Figure()
    if not history_df.empty:
        fig.add_trace(
            go.Scatter(
                x=history_df["time_h"],
                y=history_df["wake_loss_percent"],
                mode="lines+markers",
                line={"color": "#b33", "width": 2},
                hovertemplate="t=%{x:.2f} h<br>Wake-Verlust=%{y:.2f} %<extra></extra>",
                name="Wake-Verlust",
            )
        )
    fig.update_layout(
        height=270,
        margin={"l": 0, "r": 0, "t": 15, "b": 0},
        xaxis={"title": "simulierte Zeit [h]"},
        yaxis={"title": "Wake-Verlust [%]"},
        showlegend=False,
    )
    return fig
