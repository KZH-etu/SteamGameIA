"""Constantes partagées par tous les modules du projet."""
from pathlib import Path

SEED = 42
HORIZON_DAYS = 365          # succès mesuré à 1 an
T_VALUES = [0, 7, 30, 90]   # jours observés après la sortie
TOP_QUANTILE = 0.80         # top 20 % en nombre d'avis (calculé sur le TRAIN)
MIN_POS_RATIO = 0.80        # au moins 80 % d'avis positifs
SPLIT = (0.70, 0.15, 0.15)  # train / val / test, par date de sortie

# Chemins absolus : fonctionnent depuis la racine comme depuis notebooks/
ROOT = Path(__file__).resolve().parent
RAW_DIR, PROC_DIR = ROOT / "data" / "raw", ROOT / "data" / "processed"

# Colonnes mesurées après la sortie : interdites comme variables (fuite de données)
FORBIDDEN_COLS = [
    "positive", "negative", "positive_ratio", "user_reviews", "rating",
    "recommendations", "estimated_owners", "peak_ccu", "user_score",
    "metacritic_score", "price_final", "discount", "steam_deck",
    # + toutes les colonnes de temps de jeu (average/median playtime...)
]
