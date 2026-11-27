# Module split – Steam ML project (the whole project in one day)

> **Reminder of the rules (Project Brief & Governance Policy)**
> - 4 modules, **1 owner per module**, to be reported in `ai_governance/CONTRIBUTIONS.md`.
> - Everyone writes **at least 1 entry** in `PROMPT_LOG.md` **and** at least 1 in `AI_AUDIT.md` (otherwise −2 individual points).
> - At the oral (3 min per person), everyone explains **every line of their module** and says what comes from them and what comes from the AI. There are also statistical rigour questions on the **whole** project.
> - Scaler, imputer or SMOTE fitted before the train/val split = **technical grade capped at 10/20**.
> - Report **in English** (otherwise the grade is divided by 2), in LaTeX (bonus +1/+2).

The prompts are written in English so that they can be copied as is into `PROMPT_LOG.md`. They are **examples to adapt**: write down the version you actually used.

---

## 0. Reduced scope (everything is due today)

- **SteamCharts: dropped.**
- **Subset of games** to go fast, e.g. the games released between 2015 and 2021, or a sample of a few thousand `app_id`.
  ⚠️ **We sample by game** (all the reviews of the drawn games are kept), **never by review row**. Otherwise the number of reviews per game, hence the target, is distorted.
- **Data used:**
  - Kozyriev: `recommendations.csv` and `games.csv` (not `users.csv`). `games_metadata.json` (tags, description) is optional.
  - FronkonGames: `games.csv` with `usecols`, after checking that the columns are not shifted.
- **Models:** baselines (`DummyClassifier`, logistic regression) → random forest, `HistGradientBoostingClassifier` → `MLPClassifier` (out-of-course model, with a reference paper) → voting ensemble.
- **Golden rule:** a version that runs end to end (baseline and results table) **before** moving to the advanced models.

---

## 1. Common conventions (slot 0, all together, 30 min)

**Repository structure (= structure of the .zip)**
```
Group_XX_DS_Project/
├── requirements.txt          # pinned versions (pip freeze)
├── config.py                 # paths, SEED, constants
├── notebooks/
│   ├── 01_data.ipynb         # Module 1
│   ├── 02_target_split.ipynb # Module 2
│   ├── 03_features.ipynb     # Module 3
│   └── 04_models.ipynb       # Module 4
├── data/raw/  data/processed/   # not in the .zip if too large (explain how to download them)
├── report/
│   └── Report_Group_XX.pdf
└── ai_governance/
    ├── CONTRIBUTIONS.md
    ├── PROMPT_LOG.md
    └── AI_AUDIT.md
```

**`config.py`**
```python
SEED = 42
HORIZON_DAYS = 365          # success measured after 1 year
T_VALUES = [0, 7, 30, 90]   # days observed after release
TOP_QUANTILE = 0.80         # top 20% in number of reviews (computed on the TRAIN set)
MIN_POS_RATIO = 0.80        # at least 80% positive reviews
SPLIT = (0.70, 0.15, 0.15)  # train / val / test, by release date
RAW_DIR, PROC_DIR = "data/raw", "data/processed"
```

**Rules**
- Common key: **`app_id`** (integer). After each join: `assert df["app_id"].is_unique`.
- During development, everyone works on a **sample** (`nrows=...`). The final subset is only used from slot 2.
- Columns **forbidden as features** (measured after release): `positive`, `negative`, `positive_ratio`, `user_reviews`, `rating`, `recommendations`, `estimated_owners`, `peak_ccu`, playtime, `user_score`, `metacritic_score`, `price_final`, `discount`, `steam_deck`.
- **The test set is used only once**, in slot 4. All choices are made on the validation set.

**Files produced by each module**

| Module | File | Minimum columns |
|---|---|---|
| 1 | `games_clean.parquet` | `app_id`, `release_date`, cleaned columns (base price, platforms, languages, genres, tags…) |
| 1 | `reviews_clean.parquet` | `app_id`, `date`, `days` (days since release), `is_recommended`, `hours`, `helpful` |
| 2 | `target.parquet` | `app_id`, `release_date`, `n_reviews_1y`, `pos_ratio_1y`, `success`, `split` |
| 3 | `features_static.parquet` | `app_id` + features known at release |
| 3 | `features_T{7,30,90}.parquet` | `app_id` + early signals |
| 4 | `results.csv` | model, T, PR-AUC, ROC-AUC, F1, 95% CI |

**Mapping with the week 2 objectives (to copy into `CONTRIBUTIONS.md`)**

| Objective | Module(s) |
|---|---|
| Clean and preprocess the dataset | 1 (raw cleaning), 3 (transformation into features), 4 (imputation and scaling in the pipeline) |
| Analyze features, missing values and target distribution | 3 (features, missing values), 2 (target) |
| Define training, validation and test sets | 2 |
| Select and implement a baseline model | 4 |
| Evaluate initial performance using appropriate metrics | 4 (metrics), 2 (confidence intervals) |

---

## 2. Schedule of the day

| Slot | Module 1 – Data | Module 2 – Target and split | Module 3 – Features | Module 4 – Models |
|---|---|---|---|---|
| **0. Together (30 min)** | Repository, `config.py`, `requirements.txt`, file columns, `ai_governance/` folder | ← | ← | ← |
| **1. Preparation (~1 h 30)** | Cleaning and merging, **version 0 on a sample in less than one hour**, then computation of `days` | Target code on the raw sample | Code of the static features and of the T signals on the sample | Pipeline and baselines on a small table, **skeleton of the LaTeX report** |
| **2. Baseline (~1 h 30)** | Run on the final subset, check `assert`s | Split by date, threshold on the train set, target distribution | Missing values, correlations, list of excluded columns | Assembly, **baselines and first results table** |
| **3. Models (~2 h)** | `MLPClassifier` + scientific paper, voting ensemble | Class imbalance, confidence intervals, model comparison | Feature selection, feature importance for each T | Random forest, gradient boosting, tuning, **curve as a function of T** |
| **4. Finalisation (~1 h 30)** | Error analysis (types of games badly predicted) | "Uncertainty" part | Interpretation: what makes a success? | **Final evaluation on the test set, only once** |
| **Everyone, at the end of the day** | Their section of the report (in English), their `PROMPT_LOG` and `AI_AUDIT` entries, review of the next module (1→2→3→4→1) | ← | ← | ← + assembly of the .zip |

**If we run out of time:** in slot 3, we only keep the logistic regression and gradient boosting, and drop the model ensemble. The MLP stays, because an out-of-course algorithm is required.

---

## Module 1 – Data: loading, cleaning, merging, out-of-course model

**Owner:** ______________

**What is in the module**
- *Slots 1-2:*
  - Load Kozyriev (`games.csv`, `recommendations.csv`) and FronkonGames (`games.csv` with `usecols`), with compact dtypes.
  - Check the column alignment of the FronkonGames CSV, and split again the lists stored as text (tags, genres, languages).
  - Convert the dates, remove duplicates, remove impossible values.
  - Merge the two game tables on `app_id` (base = Kozyriev). Count the lost games, and compare the two release dates.
  - Attach `release_date` to the reviews and compute `days`. Count the reviews with `days < 0` (early access) and exclude them.
  - Draw the subset of games, **by `app_id`**.
- *Slot 3:* `MLPClassifier` as the out-of-course model, justified by a scientific paper; voting ensemble of the best models.
- *Slot 4:* error analysis. On which games is the model wrong (genre, price, number of reviews)?

**Prompt ideas**

1. *Loading and inspection*
   > I have `recommendations.csv` (41M rows; columns: app_id int, helpful int, funny int, date str YYYY-MM-DD, is_recommended bool, hours float, user_id int, review_id int) and `games.csv` (app_id, title, date_release, win, mac, linux, rating, positive_ratio, user_reviews, price_final, price_original, discount, steam_deck). Write pandas code that loads only app_id, date, is_recommended, hours, helpful from recommendations with compact dtypes (int32, bool, float32), parses dates, and prints shape, dtypes, % missing per column and the number of duplicated rows. Do not drop or impute anything yet. Tell me the expected memory usage and how to read the file in chunks if I have less than 12 GB of RAM.

2. *Merge and subset*
   > Merge the Kozyriev games table (base table) with the FronkonGames table (key `AppID` renamed to `app_id`) using a left join with `validate="one_to_one"` and `indicator=True`. Report the number of games in each source, how many are kept, and compare the two release dates (flag differences > 30 days). Then draw a random sample of N app_id (seed fixed) and keep ALL reviews of these games: do not sample individual review rows. Explain why sampling review rows would bias the number of reviews per game.

3. *Out-of-course model*
   > Add an MLPClassifier to the existing scikit-learn pipeline (imputer + scaler fitted on train only). Use early_stopping=True, random_state=SEED, and tune hidden_layer_sizes and alpha on the validation set only (PredefinedSplit). Compare it with the logistic regression baseline on PR-AUC. Give me one peer-reviewed reference that justifies using a neural network on tabular data, and tell me its main limitation compared to gradient boosting.

**AI mistakes to watch for**
- A merge that **duplicates** rows (`app_id` not unique).
- Badly parsed dates ("Coming soon", US format).
- Sampling **by review row** instead of by game.
- The review file loaded entirely with the default dtypes, which crashes the PC.
- An **invented scientific reference**: check on Google Scholar that the paper really exists.

**Possible oral questions:** why a `left` join? How many games are lost, and does it bias the sample? Why sample by game? Why does the MLP do better or worse than gradient boosting?

---

## Module 2 – Target, split and uncertainty

**Owner:** ______________

**What is in the module**
- *Slots 1-2:*
  - For each game, `n_reviews_1y` and `pos_ratio_1y` on the reviews with `0 ≤ days < 365`. We start from **all the games** (`left` join): a game without reviews has 0 reviews, it is not a row to delete.
  - Keep the games that have a full year of reviews.
  - Split by release date (70 / 15 / 15), then top 20% threshold computed **on the train set**, and finally `success`.
  - Target distribution per split and over time.
- *Slot 3:*
  - Class imbalance: `class_weight` compared with resampling **inside** the pipeline.
  - 95% bootstrap confidence intervals on the validation set.
  - Model comparison: is the difference significant?
- *Slot 4:* "uncertainty and limitations" part of the report.

**Prompt ideas**

1. *Target computation*
   > Using `reviews_clean` (app_id, days, is_recommended) and `games_clean` (app_id, release_date), compute for each game `n_reviews_1y` and `pos_ratio_1y` using only reviews with `0 <= days < 365`. Start from ALL games in games_clean with a left join, so that games with zero reviews get n_reviews_1y = 0 instead of disappearing. Keep only games where `release_date + 365 days <= max(review date)`. Use vectorized pandas and add an assert that no review outside the window was used.

2. *Split and label*
   > Split the games by release_date: oldest 70% = train, next 15% = validation, most recent 15% = test (no shuffling). Compute the 80th percentile of `n_reviews_1y` on the TRAIN set only and use it for all splits: `success = (n_reviews_1y >= threshold) & (pos_ratio_1y >= 0.8)`. Print the threshold, the date range and the positive rate of each split. Explain why computing the threshold on the full dataset would be data leakage.

3. *Uncertainty and comparison*
   > Given validation labels and the predicted probabilities of two models, compute a 95% bootstrap confidence interval (1000 resamples, fixed seed) for the PR-AUC of each model, and a paired bootstrap test on the PR-AUC difference (resample the same indices for both models). State the assumptions of this test and when it would not be valid.

**AI mistakes to watch for**
- A **random `train_test_split`** instead of a split by date.
- A threshold computed on **all** the data.
- Games without reviews **deleted** instead of being kept with 0 reviews: survivorship bias.
- SMOTE or resampling applied **before** the split: grade capped at 10/20.
- A non-paired bootstrap to compare two models.

**Possible oral questions:** why a time-based split? What happens in case of a distribution shift between old and recent games? Why "top 20%" and not a fixed threshold? What does a confidence interval overlapping the baseline's mean?

---

## Module 3 – Features, exploratory analysis and selection

**Owner:** ______________

**What is in the module**
- *Slots 1-2:*
  - **Static features**: log of the base price, free or not, platforms, number of languages, release month and day, description length, tags, genres and categories as binary features (tags kept according to their frequency **on the train set**).
  - **Early signals** for T = 7, 30 and 90: number of reviews, % positive, mean and median hours played, "helpful" votes. Values stay empty when there is no review, with a `has_reviews` flag.
  - Exploratory analysis: missing values per split, distributions, outliers, correlations.
  - **Leakage audit**: list of the excluded columns, with the reason.
- *Slot 3:* feature selection **inside the pipeline** (filter, L1, RFE, as in lab 4); feature importance for each T (permutation importance on the validation set).
- *Slot 4:* interpretation. Which features matter at T = 0, then at T = 30?

**Prompt ideas**

1. *Static features*
   > From these columns [price_original, win, mac, linux, Supported languages, release_date, About the game, Genres, Categories, Tags, DLC count, Required age], build features known at release time: log1p(price), is_free, platform flags, n_languages, release month and weekday, description length, and multi-hot encoding of genres/categories/tags keeping only tags present in at least 1% of TRAIN games. For each column, tell me whether it could have changed after release (e.g. DLC count, tags) and flag it as a leakage risk.

2. *Early signals*
   > For each T in [7, 30, 90], compute per game from `reviews_clean` with `0 <= days < T`: n_reviews, pos_ratio, mean and median hours, total helpful votes. Start from all games (left join): n_reviews = 0 when there is no review, pos_ratio and hours stay NaN, and add a has_reviews flag. Do not impute anything here (imputation is done in the model pipeline). Add an assert that no review with days >= T is used.

3. *Selection and importance*
   > Add feature selection INSIDE the scikit-learn pipeline (SelectKBest with mutual_info_classif, and SelectFromModel with an L1 logistic regression), tune k or C on the validation set only, and report which features are kept for T = 0 and T = 30. Then compute permutation importance on the validation set for the best model. Explain why selecting features on the full dataset before splitting would be leakage.

**AI mistakes to watch for**
- The AI suggests `positive_ratio`, `user_reviews`, `estimated_owners` or `peak_ccu` as features: **data leakage**.
- A window that uses reviews **after T**.
- Tags kept according to their frequency on **all** the data.
- Missing values replaced by 0 without saying so (0 reviews ≠ missing data).
- Feature selection done **before** the split.

**Possible oral questions:** why exclude this column? What happens if a tag was added after release? Price is correlated with success: is it a causal relationship?

---

## Module 4 – Pipeline, models and final evaluation

**Owner:** ______________

**What is in the module**
- *Slots 1-2:*
  - Assemble the files of Modules 2 and 3 on `app_id`: one dataset per value of T.
  - scikit-learn pipeline: imputation, scaling and encoding **inside** the pipeline, fitted on the train set only.
  - `DummyClassifier` and logistic regression baselines (T = 0).
  - Metrics on the validation set: **PR-AUC (main)**, ROC-AUC, F1, precision, recall, confusion matrix. Compare with the PR-AUC of a random model (= share of successes).
  - Skeleton of the LaTeX report.
- *Slot 3:* random forest, `HistGradientBoostingClassifier`, tuning with `RandomizedSearchCV` + `PredefinedSplit` (validation by date); **"PR-AUC as a function of T" curve**.
- *Slot 4:* final evaluation **only once** on the test set, comparison with the baseline, assembly of the .zip.

**Prompt ideas**

1. *Pipeline and baseline*
   > Build a scikit-learn Pipeline with a ColumnTransformer: numerical features → SimpleImputer(strategy="median") then StandardScaler; binary multi-hot features → passthrough. The pipeline must be fitted on TRAIN only. Models: DummyClassifier(strategy="most_frequent") and LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED). Evaluate on VALIDATION with average_precision_score (PR-AUC), roc_auc_score, F1, precision, recall and the confusion matrix. Also print the PR-AUC of a random classifier (= positive rate). Do not use the test set.

2. *Tuning and curve as a function of T*
   > Tune a HistGradientBoostingClassifier and a RandomForestClassifier with RandomizedSearchCV (20 iterations, random_state=SEED) using a PredefinedSplit where the fold is our time-based validation set (no random CV, because our split is temporal). Use average_precision as the scoring. Repeat for T in [0, 7, 30, 90] with the same target and split, and plot validation PR-AUC versus T for each model, with the baseline as a horizontal line.

3. *Final evaluation*
   > Refit the selected model (chosen on validation only) on train+validation, then evaluate it ONCE on the test set: PR-AUC, ROC-AUC, F1 at the threshold chosen on validation, with 95% bootstrap confidence intervals. Explain why the test score can be lower than the validation score.

**AI mistakes to watch for**
- `StandardScaler` or `SimpleImputer` fitted **before** the split: **grade capped at 10/20**.
- **Accuracy** as the main metric with imbalanced classes.
- `roc_auc_score` computed on the predicted classes instead of `predict_proba`.
- A **random** cross-validation for tuning, although the split is temporal.
- The test set used to choose the model or the threshold.
- `random_state` forgotten: different results at each run.

**Possible oral questions:** why PR-AUC rather than accuracy? What happens to the PR-AUC if the share of successes changes? What does `class_weight="balanced"` do? Why a `PredefinedSplit`?

---

## Report (in English, LaTeX)

Everyone writes the section of their module. Module 4 assembles it.

| Report section | Who |
|---|---|
| Business scope, problem formalisation | Everyone (taken from the specification) |
| Data description, cleaning, merging, sampling | Module 1 |
| Target definition, data splitting, class imbalance | Module 2 |
| Features, missing values, outliers, leakage audit, feature selection | Module 3 |
| Models, hyperparameters, results vs baseline, performance vs T | Module 4 |
| Out-of-course model (MLP) + scientific reference, error analysis | Module 1 |
| Uncertainty (confidence intervals), limitations | Module 2 |
| Discussion and conclusion | Everyone |

---

## Entry templates for the governance files

**`PROMPT_LOG.md`**
```markdown
### Entry N – <title> (Module X – <first name>)
- **Goal:** ...
- **Bare prompt (v1):** "..."  → problem with the answer: ...
- **Constrained prompt (v2):** "..." (schema, leakage boundaries, assumptions, fallback)
- **What we kept / changed:** ...
```

**`AI_AUDIT.md`**
```markdown
### Flaw N – <title> (Module X – <first name>)
- **AI output:** what the AI generated (code snippet)
- **Problem:** what is wrong + root cause (e.g. scaler fitted on full data → leakage)
- **How we detected it:** test, assert, metric that looked too good...
- **Fix:** corrected code
- **Impact:** metric before / after the fix
```
