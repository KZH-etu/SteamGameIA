# Revision – Jason ZOH

Your parts: **Module 3 – features and EDA** (`03_features_eda.ipynb`), **MLP** (`05_models.ipynb`, section 4), **feature selection and permutation importance** (`06_feature_selection.ipynb`), and the integration (`main.py`, translation, sections per owner). Read `00_common.md` first.

---

## 1. Module 3 – features and EDA (`03_features_eda.ipynb`)

### Static features (known at release, T = 0)
| Feature | Computation |
|---|---|
| `log_price` | `log(1 + base price)` (price very skewed) |
| `is_free` | base price = 0 (19% of games) |
| `win`, `mac`, `linux` | platforms |
| `required_age` | minimum age |
| `n_languages` | commas in the language list + 1; `"[]"` → 0 |
| `release_month`, `release_weekday` | categories (one-hot encoded in the model pipeline) |
| `desc_length` | description length (log in the model) |
| `in_fronkon` | game found in FronkonGames (otherwise all its tags are 0) |
| genres / categories / tags | **multi-hot**: one 0/1 column per value (a game can have several) |

- Tags/genres/categories kept only if present in **≥ 1% of the train games** (frequency computed on train only → no leakage): 14 genres, 33 categories (24 after removing 9 late ones), 167 tags. `features_static.parquet`: 217 columns.
- Multi-hot vs one-hot: one-hot = exactly one 1 per row (month); multi-hot = several 1s (genres "Action,Indie").

### Early signals (`features_T7/T30/T90.parquet`)
For each T, reviews with `days < T` (assert: none with `days >= T`), per game: `n_reviews`, `pos_ratio`, `has_reviews`. `reindex` on all games: no review → `n_reviews = 0`, `pos_ratio` **NaN** (unknown, not 0: 0 would mean "all negative"), `has_reviews = False`. Share of games with ≥ 1 review: 68.2% (T = 7), 72.9% (T = 30), 75.8% (T = 90).

### EDA results
- Missing values: FronkonGames columns empty for ~22% of games, **same rate in the three splits** (21.6–23.2%) → not linked to the release date. Nothing imputed here: median imputation happens **inside** the model pipeline (fitted on train).
- Distributions very skewed (median 12 reviews in year 1, max 71,110) → log transforms. 65 games declare 103 languages: real games, kept.
- **Spearman** correlations (on ranks, robust to extreme values; train only): `n_reviews` at T = 30 ↔ success **0.53**, `has_reviews` 0.24, `n_languages` 0.23 (best at release).

### Your AI audit entries
- **Flaw 2 – hours and helpful votes (leak).** Filtering on `days < T` restricts the **review date**, not when the other columns were measured. `hours` is the playtime at collection time: 13.6% of first-month reviews have more hours than the time since release (149 h two days after release). Removed. Measured impact: adding them back changes the baseline by −0.001 [−0.005, +0.003]: real leak, no signal.
- **Flaw 3 – `ast.literal_eval` crash.** The language `K'iche'` breaks the Python-list parsing (67 unreadable lists). Fix: count commas.
- **Flaw 5 – Steam categories added after release (leak).** Remote Play Together (created 2019) on 10.5% of 2015 games; Trading Cards granted after a sales threshold. Spotted through the baseline coefficients (`Remote Play on Tablet` +1.17). 9 categories removed; T = 0 score 0.435 → 0.411 (the honest one).

### Possible questions
- *How do you know a feature is known at T?* Column-by-column audit: "when is this value measured?" (section 5 of the notebook).
- *Why keep tags at all, if store pages change?* Most tags describe the game's content, stable over time; the clearly late categories were removed. It remains a limitation.
- *Why NaN and not 0 for `pos_ratio`?* No review ≠ all negative reviews. The pipeline imputes the median and `has_reviews` tells the model the value was missing.

---

## 2. MLP (`05_models.ipynb`, section 4) – out-of-course model

### Idea
Multilayer perceptron: input → 1 or 2 hidden layers of neurons (weighted sum + non-linear activation, ReLU) → probability. Trained by gradient descent (Adam). Scaling is required (done in the pipeline).

### Hyperparameters (random search, 8 configurations per T)
- `hidden_layer_sizes`: (32,), (64,), (128,), (64, 32), (128, 64);
- `alpha`: L2 penalty on the weights, 0.001 → **10** (widened, see below);
- `learning_rate_init`: 0.001, 0.003;
- `early_stopping=True`: 10% of the train set is held out and training stops when its score stops improving → limits overfitting.

### Results
0.448 / 0.853 / 0.895 / 0.922: better than the logistic regressions, below the tree models (gain at T = 30: +0.084 [0.060, 0.112]; not significant at T = 0). Expected: on tabular data with many 0/1 features and few rows, tree-based models usually beat neural networks (Grinsztajn et al., NeurIPS 2022; Shwartz-Ziv & Armon, 2022 – check the references before citing).

### AI audit – Flaw 8 (grid edge)
First run: best `alpha` = 1, the largest value tested, at T = 7, 30, 90. Grid widened to 10: the MLP now picks `alpha = 3` at T = 7 and 30 (**inside** the grid); 0.892 → 0.895 at T = 30, 0.915 → 0.922 at T = 90.

---

## 3. Feature selection (`06_feature_selection.ipynb`, lab 4)

### Method
The three families of lab 4, always **inside the pipeline** (preprocess → selector → baseline logistic regression), fitted on train, compared on validation, at T = 0 and T = 30:
| Family | Tool | Principle |
|---|---|---|
| Filter | `SelectKBest(mutual_info_classif)`, k ∈ {10, 20, 50, 100, all} | scores each feature alone (mutual information with `success`: catches non-linear links) |
| Embedded | `SelectFromModel` + L1 logistic regression, C ∈ {0.01 … 1} | the L1 penalty sets coefficients to exactly 0 (soft-thresholding); smaller C → fewer features |
| Wrapper | `RFE`, k ∈ {10, 20, 50, 100}, step 10 | trains the model, removes the 10 weakest features, repeats; slower, takes interactions into account |

Why inside the pipeline: selecting on all the data before the split lets the validation games influence the choice → optimistic score (lab 4, Q5).

### Results
- **T = 30**: every family beats the baseline with all 236 features (0.810). About 10 features already give ≈ 0.84. **Best: L1 with 33 features, 0.852.** → the baseline overfits its many sparse 0/1 features; consistent with the grid search of notebook 05 choosing an L1 penalty.
- **T = 0**: noisy. L1 with 67 features 0.466, RFE 50–100 features ≈ 0.43, baseline 0.407; the mutual information filter never beats it (probably because it scores features one by one). These gaps are inside the baseline interval [0.34, 0.47].
- Strongest L1 coefficients: T = 30 → `n_reviews` 2.93, `pos_ratio` 2.33, then `tag_Open World` −0.65; T = 0 → `is_free` 1.42, Singleplayer, Steam Cloud, Hentai, Great Soundtrack, Steam Workshop, Visual Novel, Anime.

### Permutation importance (gradient boosting, validation)
Shuffle one feature among the validation games, measure the PR-AUC drop (10 repeats). Works for any model and measures what generalises.
- **T = 30**: `n_reviews` ≈ 0.71, `pos_ratio` ≈ 0.27, everything else < 0.01.
- **T = 0**: `n_languages` ≈ 0.06, Steam Cloud ≈ 0.04, `log_price` ≈ 0.03, then Singleplayer, Hentai, Great Soundtrack; wide error bars.
- `is_free` (top L1 coefficient) is absent: it duplicates `log_price` (price 0). With correlated features, shuffling one costs little since the model still has the other.

### Answer to the research question
At release: reach (languages, platforms, Steam Cloud/Workshop), price (free or not) and a few niche tags, but only PR-AUC ≈ 0.4–0.5. After one month: the number and tone of the first reviews carry almost all the signal.

### Possible questions
- *Coefficients vs permutation importance?* Coefficients: linear model only, need scaled features, split importance between correlated features. Permutation: any model, on validation, but also underestimates correlated features.
- *Why not select on the test set?* Same as any choice: validation only.
- *Why does selection help a regularised model?* The baseline uses C = 1 (weak regularisation): with ~236 columns and only 924 positive training games it overfits; removing features is another form of regularisation.
