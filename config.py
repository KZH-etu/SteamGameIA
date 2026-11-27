"""Constants shared by all the modules of the project."""
from pathlib import Path

SEED = 42
HORIZON_DAYS = 365          # success measured after 1 year
T_VALUES = [0, 7, 30, 90]   # days observed after release
TOP_QUANTILE = 0.80         # top 20% in number of reviews (computed on the TRAIN set)
MIN_POS_RATIO = 0.80        # at least 80% positive reviews
SPLIT = (0.70, 0.15, 0.15)  # train / val / test, by release date

# Absolute paths: work from the project root as well as from notebooks/
ROOT = Path(__file__).resolve().parent
RAW_DIR, PROC_DIR = ROOT / "data" / "raw", ROOT / "data" / "processed"

# Columns measured after release: forbidden as features (data leakage)
FORBIDDEN_COLS = [
    "positive", "negative", "positive_ratio", "user_reviews", "rating",
    "recommendations", "estimated_owners", "peak_ccu", "user_score",
    "metacritic_score", "price_final", "discount", "steam_deck",
    # + every playtime column (average/median playtime...)
]
