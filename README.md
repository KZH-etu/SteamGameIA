# SteamGameIA – Predicting the success of a Steam game

Predict whether a Steam game will be a success one year after its release, i.e. whether it is in the top 20% for the number of reviews **and** has at least 80% positive reviews. We use the information known at release and the first signals observed at T = 0, 7, 30 and 90 days.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data
Put the source files in `data/raw/` (not versioned):
- FronkonGames – Steam Games Dataset
- Kozyriev – `games.csv`, `recommendations.csv`, `games_metadata.json`

## Structure
| Folder / file | Content |
|---|---|
| `config.py` | Paths, SEED, constants |
| `main.py` | Reruns the notebooks in order (01 → 05) |
| `notebooks/01_data.ipynb` | Module 1 – loading, cleaning, merging |
| `notebooks/02_target_split.ipynb` | Module 2 – target and time-based split |
| `notebooks/03_features_eda.ipynb` | Module 3 – features and EDA |
| `notebooks/04_baseline.ipynb` | Module 4 – baseline and evaluation |
| `notebooks/05_models.ipynb` | Module 4 (step 3) – grid search, random forest, gradient boosting, MLP, voting / bagging / stacking, bootstrap CIs |
| `report/` | Final report |
| `ai_governance/` | CONTRIBUTIONS, PROMPT_LOG, AI_AUDIT |

See `MODULES.md` for the work split and the rules.

## Rerun the whole pipeline
Each notebook reads the files written by the previous ones in `data/processed/` (01 → 02 → 03 → 04).
After a change, rerun the modified notebook **and all the ones after it**:
```bash
python main.py       # rerun everything, from 01 to 04 (01 needs data/raw/)
python main.py 03    # rerun from 03 (03 then 04)
```
The notebooks are rewritten with their new results. The script stops at the first notebook that fails.

## Git workflow
Each owner works on **their own branch** and never pushes directly to `main`.

| Branch | Module |
|---|---|
| `module1-data` | 1 – Data |
| `module2-target` | 2 – Target and split |
| `module3-features` | 3 – Features and EDA |
| `module4-baseline` | 4 – Baseline and evaluation |

```bash
git switch module1-data            # move to your branch
git pull origin main               # get the latest merges (before starting)
# ... work, regular commits ...
git push -u origin module1-data
```
When the module is ready: open a Pull Request to `main`, have it reviewed by another member, then merge.
The modules depend on each other (1 → 2 → 3 → 4): merge in this order and run `git pull origin main` on your branch after each merge.
To limit conflicts, everyone only edits **their own notebook**; changes to `config.py` or `requirements.txt` are announced to the group. In `ai_governance/`, everyone adds their entries at the end of the file, which limits conflicts.
