# Revision – Yohan ALLOUCHE

Your parts: **Module 1 – data** (`01_data.ipynb`), **gradient boosting** (`05_models.ipynb`, section 3) and **error analysis** (`07_error_analysis.ipynb`). Read `00_common.md` first.

---

## 1. Module 1 – loading and cleaning (`01_data.ipynb`)

### What the notebook does, step by step
1. **Reviews inspection** on a development sample (`NROWS = 5_000_000` of the 41 M rows): compact dtypes (`int32`, `bool`, `float32`), dates parsed with `errors="coerce"` (unreadable → `NaT`, counted, not dropped). 185,657 "duplicated rows" are **not** removed: two players can post the same values on the same day.
2. **Kozyriev `games.csv`**: 50,872 games, no missing value, no duplicated `app_id`.
3. **FronkonGames `games_1.csv`**: the header is broken. `DiscountDLC count` is one name for two columns, so every column after `Price` is shifted. We rebuild the column list (`cols[i:i+1] = ["Discount", "DLC count"]`) and read with `names=cols, skiprows=1`. Without the fix, no game matched; with it, 77.5% of Kozyriev games are found.
4. **Game subset**: games released 2015–2021, then `sample(10_000, random_state=SEED)`. We sample **games, never review rows**: sampling reviews would change each game's number of reviews, hence the target.
5. **`reviews_clean`**: the 2 GB file is read in chunks of 2 M rows, and **each chunk is filtered** on the sampled games before being kept. 7,304,209 reviews, 7,906 games with at least one review.
6. **`days` = review date − release date**. 470,961 reviews (6.45%) have `days < 0` (posted before the official release, early access): removed, because target and signals start at release. 6,833,248 reviews kept.
7. **`games_clean`**: `subset.merge(fronkon, how="left", validate="one_to_one", indicator=True)`.
   - `left`: every sampled game is kept, even if missing from FronkonGames (7,797 found, 2,203 missing).
   - `validate="one_to_one"`: error if an `app_id` is duplicated (would duplicate rows).
   - `indicator=True`: tells which games were found.
   - 143 games have release dates differing by more than 30 days between the two sources; we keep the Kozyriev date.
8. **Columns removed because measured after release (leakage):** `rating`, `positive_ratio`, `user_reviews`, `price_final`, `discount`, `steam_deck`, FronkonGames `Price`.

### Your AI audit entry – Flaw 1 (memory crash)
The AI read the file "in chunks" but kept **every** chunk and concatenated them (41 M rows in memory), then ran `duplicated()` on the whole table: the kernel died after 82 s. Fix: a development sample for the inspection, and filtering each chunk **inside** the loop. Lesson: reading in chunks only saves memory if each chunk is reduced before being kept.

### Possible questions
- *Why 10,000 games and not all of them?* Computation time on our laptops; the sample is random (seed 42) and by game, so the target is not distorted. `N_GAMES = None` would take all 35,588 games of 2015–2021.
- *Why remove early-access reviews?* The target and the early signals count from the official release; a game with a long early access would otherwise start with many reviews.
- *Why left join?* An inner join would drop the 2,203 games absent from FronkonGames, and nothing guarantees that they are a random subset; an `in_fronkon` flag (Module 3) lets the model know which games have no store data.
- *Why is FronkonGames `Price` excluded but Kozyriev `price_original` kept?* `Price` is the price at collection time (may include later changes); `price_original` is the base price.

---

## 2. Gradient boosting (`05_models.ipynb`, section 3)

### Idea
`HistGradientBoostingClassifier`: trees are added **one after the other**, each new small tree corrects the errors (the gradient of the loss) of the current model. Boosting mainly **reduces the bias**; the random forest averages independent deep trees and mainly **reduces the variance**. "Hist" = features are grouped into bins (histograms), which makes it fast; it also handles missing values natively.

### Hyperparameters searched (random search, 8 configurations per T)
| Parameter | Meaning | Values |
|---|---|---|
| `learning_rate` | weight of each new tree (small = careful) | 0.03, 0.05, 0.1 |
| `max_leaf_nodes` | size of each tree | 15, 31, 63 |
| `min_samples_leaf` | minimum games per leaf | 20, 50, 100 |
| `l2_regularization` | penalty on leaf values | 0, 0.1, 1, **10** (widened) |
| `max_iter` | number of trees | 200, 400 |
| `class_weight` | weight of the rare class | None, balanced |

Chosen at T = 30: learning rate 0.03, 31 leaves, 100 games per leaf, 400 trees, `l2_regularization = 10`, balanced.

### Results (validation PR-AUC)
0.483 / **0.875** / **0.917** / **0.946** at T = 0 / 7 / 30 / 90. Best model at T = 7 and 30, tied with stacking at T = 90. Gain over the baseline at T = 30: +0.107, 95% CI [0.076, 0.142]. It is our **proposed final model** (as good as stacking, about 6× faster, simpler).

### AI audit – Flaw 8 (grid edge)
In the first run the best `l2_regularization` was 1, the largest value tested: maybe a stronger value is better. The grid was widened to 10 → 0.915 → 0.917 at T = 30 (within noise). It now picks 10, **again the edge**: a limitation to mention.

### Possible questions
- *Boosting vs random forest?* Forest: deep independent trees on bootstrap samples, averaged (variance ↓). Boosting: small trees in sequence, each fixing the previous errors (bias ↓); more sensitive to the learning rate and number of trees.
- *Why random search?* For the same budget it tries more different values of each parameter than a grid (Bergstra & Bengio, 2012). 8 configurations is small: a limitation.
- *Why F1 = 0.786 only, with the best PR-AUC?* F1 uses the 0.5 threshold; the threshold should be chosen on validation (week 4).

---

## 3. Error analysis (`07_error_analysis.ipynb`)

### What the notebook does
- Reads the **validation probabilities** saved by notebook 05 (`val_proba.parquet`): the model is **not retrained**, so we analyse exactly the model of the comparison table. The test set is not used.
- Merges them with `target.parquet` (true `n_reviews_1y`, `pos_ratio_1y`), `features_T30` and `games_clean` (title, price, genres).
- Threshold 0.5 → each game is TP, FP (false alarm), FN (missed success) or TN with `np.select`.

### Results (gradient boosting, T = 30)
- **208 TP, 86 FP, 27 FN, 1,178 TN.**
- **Errors are at the threshold of the target:** no error on the 324 games without any review in year 1, 2.3% on games with 1–32 reviews, **31.8%** between 33 and 65 reviews (41 FP), 20.4% between 66 and 131. Part of the error comes from the hard 66-review threshold: 60 and 70 reviews are very similar games.
- **False alarms:** 57 do not reach 66 reviews; 29 reach it but end below 80% positive. Their share of positive reviews drops on average from 0.895 (T = 30) to 0.851 (one year): games criticised later, invisible at T = 30.
- **Missed successes start slowly:** median 17 reviews in the first month (115.5 for TP), 77% positive (93% for TP). Several are free games whose rating rose over the year (e.g. 0.675 → 0.807).
- **Free games are the hardest:** 10.0% errors vs 8.1% (price ≥ 10) and 6.2% (< 10), although free and ≥ 10 games have similar success rates (21.1% vs 22.4%). By genre: Free To Play ≈ 12% errors, Early Access ≈ 5%.
- **Limitations:** one model, one T, one threshold; genres overlap and groups are small (≥ 50 games), so rates are noisy.

### Possible questions
- *Is the model bad on these games, or is the target hard?* Mostly the target: errors concentrate where the label is decided by a few reviews or a few percent of positive reviews.
- *How could we reduce errors?* A softer target (regression on the number of reviews, or a margin around the threshold), signals on the trend of reviews (week 1 vs week 4), choosing the threshold on validation.
- *Why compare error rate with success rate?* Errors are more frequent where classes are mixed; a group with more successes naturally has more errors. Free games have more errors than paid games with the same success rate, so they are genuinely harder.
