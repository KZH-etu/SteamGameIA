# SteamGameIA – Prédiction du succès d'un jeu Steam

Prédire si un jeu Steam sera un succès un an après sa sortie, c'est-à-dire s'il est dans le top 20 % en nombre d'avis **et** a au moins 80 % d'avis positifs. On utilise les informations connues à la sortie et les premiers signaux observés à T = 0, 7, 30 et 90 jours.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Données
Placer les fichiers sources dans `data/raw/` (non versionnés) :
- FronkonGames – Steam Games Dataset
- Kozyriev – `games.csv`, `recommendations.csv`, `games_metadata.json`

## Structure
| Dossier / fichier | Contenu |
|---|---|
| `config.py` | Chemins, SEED, constantes |
| `notebooks/01_data.ipynb` | Module 1 – chargement, nettoyage, fusion |
| `notebooks/02_target_split.ipynb` | Module 2 – cible et découpage temporel |
| `notebooks/03_features_eda.ipynb` | Module 3 – variables et EDA |
| `notebooks/04_baseline.ipynb` | Module 4 – baseline et évaluation |
| `report/` | Rapport final |
| `ai_governance/` | CONTRIBUTIONS, PROMPT_LOG, AI_AUDIT |

Voir `REPARTITION_MODULES.md` pour la répartition et les règles.
