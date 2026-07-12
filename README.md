# FOXES Student Wake GUI

Eine kleine Streamlit-Oberflaeche fuer Lehrbeispiele mit FOXES.

## Ziel

Der erste Schritt ist bewusst klein:

- Windpark als einfache Turbinenreihe oder FOXES-Testpark
- ein einzelner Windzustand mit Windgeschwindigkeit, Windrichtung, TI und Luftdichte
- Auswahl eines Turbinentyps und Wake-Modells
- Ausgabe von Leistung, Umgebungleistung und Wirkungsgrad
- Plot des Layouts

Darauf aufbauend koennen wir im naechsten Schritt Wake-Feld-Plots, Windrosen und Optimierung hinzufuegen.

## Python

FOXES unterstuetzt aktuell Python `>=3.9,<3.14`. Falls `python --version` bei dir `3.14` zeigt, installiere bitte zunaechst Python 3.12 oder 3.13.

## Installation

Im Ordner `student_wake_gui`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Falls Python 3.13 installiert ist, geht entsprechend:

```powershell
py -3.13 -m venv .venv
```

## Start

```powershell
streamlit run app.py
```

## Projektstruktur

Die App ist auf mehrere Module aufgeteilt (vorher: eine einzelne ~2200-Zeilen app.py):

- `app.py` - Einstiegspunkt: Seiten-Setup, Sidebar-Dispatch, Tab-Weiterleitung
- `foxes_runner.py` - FOXES-Simulation (Turbinenreihe/Windpark, Windklima-AEP)
- `dynamic_runner.py` - experimentelles dynamisches Wake-Paket-Modell
- `figures.py` - alle Plotly-Diagramme
- `state.py` - Session-State-Helfer (u.a. `ensure_result` fuer das
  "berechnen, falls noetig"-Muster)
- `runner_common.py` - geteilte Rampen-Mathematik fuer beide Runner
- `sidebar.py` - Sidebar-Widgets pro Arbeitsbereich
- `climate_tab.py` - Tab "Turbine & Windklima"
- `simulation_tab.py` - Tabs "Turbinenreihe" / "Windpark"
- `runner_ui.py` - Tab "Runner" (quasi-statisch)
- `dynamic_runner_ui.py` - Tab "Runner" -> "Dynamisch experimentell"
- `styles.py` - CSS

### Performance-Hinweise

- `available_turbines()`, `create_power_ct_curve()` und der FOXES-Turbinentyp
  sind mit `st.cache_data`/`st.cache_resource` gecacht.
- Wake-Streamline-/Konturlaenge werden nur noch berechnet, wenn sie auch
  angezeigt werden (Turbinenreihe-Layout + aktive Checkbox), statt immer
  mitzulaufen.
- Die Konturextraktion nutzt `contourpy` direkt statt eine komplette
  Matplotlib-Figure aufzubauen; die Konturflaechen-Suche nutzt
  `scipy.ndimage` statt eines manuellen Python-Flood-Fills.
- Der Dynamic Runner berechnet Wake-Defizite vektorisiert (NumPy-Broadcasting)
  statt in einer Python-Schleife pro Wake-Paket und Abfragepunkt.
- Beide Runner-Loops laufen ueber `st.fragment(run_every=...)` statt ueber
  `time.sleep()` + `st.rerun()` - dadurch wird bei jedem Animationsschritt
  nur der betroffene Seitenbereich neu berechnet, nicht das gesamte Skript.

