"""Constantes partagées par tous les modules du projet."""

SEED = 42
HORIZON_DAYS = 365          # succès mesuré à 1 an
T_VALUES = [0, 7, 30, 90]   # jours observés après la sortie
TOP_QUANTILE = 0.80         # top 20 % en nombre d'avis
MIN_POS_RATIO = 0.80        # au moins 80 % d'avis positifs
RAW_DIR, PROC_DIR = "data/raw", "data/processed"

# Colonnes mesurées après la sortie : interdites comme variables (fuite de données)
FORBIDDEN_COLS = [
    "positive", "negative", "positive_ratio", "user_reviews", "rating",
    "recommendations", "estimated_owners", "peak_ccu", "user_score",
    "metacritic_score", "price_final", "discount",
    # + toutes les colonnes de temps de jeu (average/median playtime...)
]
