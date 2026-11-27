"""Relance tout le pipeline : exécute les notebooks dans l'ordre (01 -> 04).

Chaque notebook lit les fichiers écrits par les précédents dans data/processed/,
donc une modification dans un notebook oblige à relancer tous ceux qui suivent.

Usage :
    python main.py       # tout relancer, de 01 à 04
    python main.py 03    # relancer à partir de 03 (03 puis 04)
"""
import subprocess
import sys

from config import ROOT

NOTEBOOKS = ["01_data", "02_target_split", "03_features_eda", "04_baseline"]

start = sys.argv[1] if len(sys.argv) > 1 else "01"
todo = [nb for nb in NOTEBOOKS if nb[:2] >= start]
if not todo:
    sys.exit(f"Aucun notebook à partir de {start} (choix : 01 à {NOTEBOOKS[-1][:2]})")

for nb in todo:
    print(f"Exécution de {nb}.ipynb ...")
    # --inplace : le notebook est réécrit avec ses nouveaux résultats et graphiques
    # timeout=-1 : pas de limite de temps par cellule (01 lit 2 Go d'avis)
    result = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                             "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1",
                             str(ROOT / "notebooks" / f"{nb}.ipynb")])
    if result.returncode != 0:
        sys.exit(f"Échec dans {nb}.ipynb : les notebooks suivants n'ont pas été relancés.")

print("Pipeline terminé : data/processed/ et report/results.csv sont à jour.")
