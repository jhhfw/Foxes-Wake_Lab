#!/usr/bin/env python3
"""Deploy-feste FOXES-Datenpfade.

Ersetzt in foxes_runner.py und dynamic_runner.py die hartkodierten, repo-relativen Datenpfade
(.../foxes/foxes/data/...) durch eine Lookup-Funktion, die die FOXES-Statikdaten im
*installierten* foxes-Paket findet - mit Fallback auf das bisherige Layout. Damit laeuft die App
auch ohne den Schwesterordner "foxes/" (Streamlit Cloud, JupyterHub, ...).

Ausfuehren im Ordner student_wake_gui:

    python apply_patch.py

Es werden .bak-Backups angelegt. Das Skript ist idempotent (mehrfaches Ausfuehren schadet nicht).
Rueckgaengig: die .bak-Datei zurueckkopieren (z. B. `copy foxes_runner.py.bak foxes_runner.py`).
"""
from pathlib import Path

OLD_FR = (
    "def _power_ct_curve_dir() -> Path:\n"
    "    return Path(__file__).resolve().parents[1] / \"foxes\" / \"foxes\" / \"data\" / \"power_ct_curves\""
)
NEW_FR = (
    "def _foxes_data_dir(subdir: str) -> Path:\n"
    "    \"\"\"Verzeichnis der FOXES-Statikdaten - deploy-fest ueber das installierte foxes-Paket.\n"
    "\n"
    "    Vorher wurde ein hartkodierter, repo-relativer Pfad (.../foxes/foxes/data/...) verwendet,\n"
    "    der nur funktioniert, wenn das FOXES-Repo als Schwesterordner daneben liegt. Beim Hosting\n"
    "    (Streamlit Cloud, JupyterHub) ist das nicht der Fall. Diese Variante findet die Daten im\n"
    "    installierten foxes-Paket und faellt nur zur Not auf das alte Layout zurueck.\n"
    "    \"\"\"\n"
    "    try:\n"
    "        import importlib.util\n"
    "        spec = importlib.util.find_spec(\"foxes\")\n"
    "        if spec is not None and spec.origin:\n"
    "            cand = Path(spec.origin).resolve().parent / \"data\" / subdir\n"
    "            if cand.exists():\n"
    "                return cand\n"
    "    except Exception:\n"
    "        pass\n"
    "    return Path(__file__).resolve().parents[1] / \"foxes\" / \"foxes\" / \"data\" / subdir\n"
    "\n"
    "\n"
    "def _power_ct_curve_dir() -> Path:\n"
    "    return _foxes_data_dir(\"power_ct_curves\")"
)

OLD_DR = (
    "def _farm_file_path():\n"
    "    return Path(__file__).resolve().parents[1] / \"foxes\" / \"foxes\" / \"data\" / \"farms\" / \"test_farm_67.csv\"\n"
    "\n"
    "\n"
    "def _power_ct_curve_dir():\n"
    "    return Path(__file__).resolve().parents[1] / \"foxes\" / \"foxes\" / \"data\" / \"power_ct_curves\""
)
NEW_DR = (
    "def _foxes_data_dir(subdir):\n"
    "    \"\"\"Deploy-feste FOXES-Statikdaten (analog zu foxes_runner._foxes_data_dir).\"\"\"\n"
    "    try:\n"
    "        import importlib.util\n"
    "        spec = importlib.util.find_spec(\"foxes\")\n"
    "        if spec is not None and spec.origin:\n"
    "            cand = Path(spec.origin).resolve().parent / \"data\" / subdir\n"
    "            if cand.exists():\n"
    "                return cand\n"
    "    except Exception:\n"
    "        pass\n"
    "    return Path(__file__).resolve().parents[1] / \"foxes\" / \"foxes\" / \"data\" / subdir\n"
    "\n"
    "\n"
    "def _farm_file_path():\n"
    "    return _foxes_data_dir(\"farms\") / \"test_farm_67.csv\"\n"
    "\n"
    "\n"
    "def _power_ct_curve_dir():\n"
    "    return _foxes_data_dir(\"power_ct_curves\")"
)

PATCHES = {
    "foxes_runner.py": (OLD_FR, NEW_FR),
    "dynamic_runner.py": (OLD_DR, NEW_DR),
}


def main():
    here = Path(__file__).resolve().parent
    for fname, (old, new) in PATCHES.items():
        fp = here / fname
        if not fp.exists():
            print(f"!  {fname}: nicht gefunden - bitte im Ordner student_wake_gui ausfuehren.")
            continue
        text = fp.read_text(encoding="utf-8")
        if "_foxes_data_dir" in text:
            print(f"=  {fname}: bereits gepatcht - uebersprungen.")
            continue
        if old not in text:
            print(f"!  {fname}: erwarteter Codeblock nicht gefunden - bitte manuell anpassen.")
            continue
        fp.with_suffix(fp.suffix + ".bak").write_text(text, encoding="utf-8")
        fp.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"+  {fname}: gepatcht (Backup: {fname}.bak).")
    print("Fertig.")


if __name__ == "__main__":
    main()
