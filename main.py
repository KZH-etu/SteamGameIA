"""Rerun the whole pipeline: execute the notebooks in order (01 -> 04).

Each notebook reads the files written by the previous ones in data/processed/,
so a change in one notebook requires rerunning all the notebooks after it.

Usage:
    python main.py       # rerun everything, from 01 to 04
    python main.py 03    # rerun from 03 (03 then 04)
"""
import subprocess
import sys

from config import ROOT

NOTEBOOKS = ["01_data", "02_target_split", "03_features_eda", "04_baseline"]

start = sys.argv[1] if len(sys.argv) > 1 else "01"
todo = [nb for nb in NOTEBOOKS if nb[:2] >= start]
if not todo:
    sys.exit(f"No notebook from {start} (choose 01 to {NOTEBOOKS[-1][:2]})")

for nb in todo:
    print(f"Running {nb}.ipynb ...")
    # --inplace: the notebook is rewritten with its new results and plots
    # timeout=-1: no time limit per cell (01 reads 2 GB of reviews)
    result = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                             "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1",
                             str(ROOT / "notebooks" / f"{nb}.ipynb")])
    if result.returncode != 0:
        sys.exit(f"{nb}.ipynb failed: the next notebooks were not rerun.")

print("Pipeline done: data/processed/ and report/results.csv are up to date.")
