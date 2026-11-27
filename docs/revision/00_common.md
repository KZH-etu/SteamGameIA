# Revision – what everyone must know

Read this first, then the file with your name. Everything here can be asked to any member at the oral.

## 1. The project in 5 sentences

1. **Question:** using what is known T days after release (T = 0, 7, 30, 90), can we predict whether a Steam game will be a **success one year after release**, and which features matter the most?
2. **Success** = in the **top 20%** of games by number of reviews in the first year (threshold **66 reviews**, computed on the train set only) **and** at least **80% positive** reviews.
3. **Data:** 10,000 games released 2015–2021 (Kozyriev reviews + FronkonGames store data), 6.8 M reviews.
4. **Split by release date:** train 70% (7,007 games, 2015-01-01 → 2020-05-20), validation 15% (1,499, → 2021-03-29), test 15% (1,494, → 2021-12-31). Success rate 13.2% / 15.7% / 14.9%.
5. **Result so far (validation):** baseline logistic regression PR-AUC 0.407 / 0.743 / 0.810 / 0.850 at T = 0 / 7 / 30 / 90; best model gradient boosting 0.483 / 0.875 / 0.917 / 0.946. The test set has **not** been used yet (week 4).

## 2. Pipeline of the code

| Notebook | Owner (week 2) | Produces |
|---|---|---|
| `01_data` | Yohan | `reviews_clean.parquet`, `games_clean.parquet` |
| `02_target_split` | Arthur | `target.parquet` (n_reviews_1y, pos_ratio_1y, success, split) |
| `03_features_eda` | Jason | `features_static.parquet`, `features_T7/T30/T90.parquet` |
| `04_baseline` | Lucas | `report/results.csv` (Dummy + logistic regression) |
| `05_models` | all (week 3) | `results_models.csv`, figures, `val_proba.parquet` |
| `06_feature_selection` | Jason (week 3) | `results_feature_selection.csv`, 2 figures |
| `07_error_analysis` | Yohan (week 3) | `errors_by_group.png` |

`python main.py` reruns all notebooks in order; `python main.py 05` reruns from 05. `config.py` holds the shared constants (SEED = 42, HORIZON_DAYS = 365, T_VALUES, TOP_QUANTILE = 0.80, MIN_POS_RATIO = 0.80, SPLIT).

## 3. Key concepts (you must be able to explain them)

**Data leakage.** Using, during training, information that would not be available at prediction time. Examples we avoided:
- columns measured after release (rating, current price, hours played, helpful votes, Steam categories added later);
- statistics computed on all games instead of the train set (success threshold, tag frequencies, imputer median, scaler mean);
- selecting features on all data before the split;
- a random split (2021 games in train, 2016 games in test).

**Why a time-based split?** In real use, the model predicts *future* games from *past* games. A random split lets the model see the future (trends, new tags) and gives too optimistic a score.

**Why the test set is used only once.** Every choice made by looking at a score fits that score a little. If we chose the model on the test set, the test score would no longer be an honest estimate. All choices are made on validation; the test set gives the final, unbiased number.

**Why PR-AUC and not accuracy.** Only ~15% of games are successes: always answering "failure" gives 84% accuracy and learns nothing. PR-AUC (area under the precision–recall curve, `average_precision_score`) only looks at how well the successes are ranked at the top. A random model scores the share of positives (**0.157** on validation). ROC-AUC is also reported but is optimistic with rare positives (it rewards ranking the many easy failures).

**Precision / recall / F1.** Precision = among games predicted "success", share that really are. Recall = among real successes, share that we found. F1 = harmonic mean. They depend on a decision threshold (we use 0.5); PR-AUC does not.

**Pipeline.** `make_pipeline(preprocess, model)`: the imputer (median), the scaler (`StandardScaler`) and the `OneHotEncoder` (month, weekday) are **fitted on the train set only**, inside the pipeline, so validation and test never influence them.

**Bias / variance.** Too simple a model (strong regularisation, shallow tree) → bias: bad on train and validation. Too complex (weak regularisation, deep tree) → variance: excellent on train, bad on validation (overfitting). Hyperparameters are chosen where the validation score is best.

**Hyperparameter tuning on our validation set.** `PredefinedSplit` with train = −1 and validation = 0 tells `GridSearchCV` / `RandomizedSearchCV` to always train on train and score on validation (not random folds). `refit=False` then manual refit on train only, because `refit=True` would retrain on train + validation.

**Bootstrap confidence intervals.** Draw 1,499 validation games *with replacement*, recompute the PR-AUC, repeat 1,000 times; the 2.5% and 97.5% percentiles give the 95% CI. **Paired** comparison: both models are scored on the *same* resampled games, and we look at the CI of the difference; if it does not contain 0, the gain is significant.

## 4. Numbers to know by heart

| | |
|---|---|
| Games / reviews | 10,000 games, 6,833,248 reviews (6.45% posted before release removed) |
| Games without any review in year 1 | 2,200 (22%), kept as failures |
| Success threshold | 66 reviews (69 if computed on all games: 36 labels would change) |
| Random PR-AUC (validation) | 0.157 |
| Baseline LR | 0.407 / 0.743 / 0.810 / 0.850 |
| Tuned LR | 0.456 / 0.821 / 0.862 / 0.885 |
| Gradient boosting | 0.483 / 0.875 / 0.917 / 0.946 |
| Biggest jump | T = 0 → T = 7 (one week of reviews) |
| Most important features at T = 30 | number of early reviews, then their share of positive reviews |

## 5. Questions any of us can get

- **Why is T = 0 so much worse?** At release we only have the store page (price, languages, tags…). The first reviews are a direct measure of popularity and satisfaction. Part of the gain is also mechanical: the first reviews are counted in the one-year target (not leakage, since they are known at T).
- **Why not accuracy?** See above: 84% for a model that never predicts a success.
- **Is 0.917 too good to be true?** We checked for leakage (week 2 audit: 3 leaks removed). The score is high because early reviews are strongly linked to reviews after one year. The final check will be the single test evaluation.
- **Which model do you recommend?** Gradient boosting: best or tied at T ≥ 7, faster and simpler than stacking. For explanations, the logistic regression and the depth-3 tree.
- **What did the AI do and how did you check it?** We used an AI assistant to write code; every answer was checked and 9 errors were found and documented in `ai_governance/AI_AUDIT.md` (e.g. memory crash, leaks, same-day split, bagging around the pipeline, bagged SVM votes). Prompts are in `PROMPT_LOG.md`.
