# Revision – Lucas GREZES

Your parts: **Module 4 – pipeline and baseline** (`04_baseline.ipynb`), **grid search of the logistic regression** (`05_models.ipynb`, section 1), **SVM, bagging, voting and stacking** (section 5), **figures, hours leak and test evaluation** (sections 7–9). Read `00_common.md` first.

---

## 1. Module 4 – pipeline and baseline (`04_baseline.ipynb`)

### What the notebook does
1. **One dataset per T** (`make_dataset(T)`): target + static features (+ `features_T{T}` if T > 0). `desc_length` and `n_reviews` are log-transformed: a row-by-row transformation that learns nothing from the data, so no leakage even before the split.
2. **Pipeline** (`make_model`):
   ```python
   make_column_transformer(
       (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM_COLS),
       (OneHotEncoder(handle_unknown="ignore"), CAT_COLS),   # month, weekday
       remainder="passthrough")                              # 0/1 columns unchanged
   ```
   Everything is **fitted on the train set only**: fitting the imputer/scaler on all data would leak validation statistics. `handle_unknown="ignore"`: a category unseen in train gets all zeros instead of an error.
3. **Two baselines on validation**:
   - `DummyClassifier(strategy="most_frequent")`: always "failure" → PR-AUC 0.157 (= share of successes), F1 = 0. The floor.
   - `LogisticRegression(max_iter=1000, class_weight="balanced")`: `balanced` gives the rare successes more weight so the model does not ignore them.
4. **Metrics** on `predict_proba`: PR-AUC (main), ROC-AUC; F1, precision, recall at threshold 0.5 (`zero_division=0` for the Dummy).
5. **Results** (PR-AUC): 0.407 / 0.743 / 0.810 / 0.850. At T = 0: 2.6× random, recall 61% (144/235) but 332 false alarms (precision 30%).
6. **Coefficients at T = 0** (scaled features → comparable): niche tags with an enthusiastic audience push to success (Visual Novel, Detective, Hentai); they also revealed the late Steam categories leak (Flaw 5).

### Why the logistic regression is the baseline
Clearly beats the Dummy at every T; handles the imbalance (`class_weight`); gives probabilities; fast; few hyperparameters; **interpretable** (coefficients). It is the reference every week-3 model must beat.

### Your AI audit entries
- **Flaw 4 – month and weekday as numbers.** The AI put them in `NUM_COLS`: a linear effect from January (1) to December (12) makes no sense. Fix: `OneHotEncoder`. No measurable gain (< 0.01), but correct encoding and interpretable per month. Lesson: a fix can be right without improving the metric.
- **Flaw 7 – bagging around the full pipeline + nested parallelism.** `BaggingClassifier` turns X into a numpy array → the `ColumnTransformer` (columns by name) crashed. Fix: preprocessing **before** the bagging step. And `n_jobs=-1` both in the forest and in the search (16 × 16 threads) never finished → forest `n_jobs=1`.

---

## 2. Grid search of the logistic regression (`05_models.ipynb`, section 1)

### Protocol (applies to all week-3 models – know it perfectly)
```python
fold = np.r_[np.full(len(Xtr), -1), np.zeros(len(Xva))]   # -1: always train, 0: the validation fold
GridSearchCV(pipe, grid, scoring="average_precision", cv=PredefinedSplit(fold), refit=False)
best = clone(pipe).set_params(**search.best_params_).fit(Xtr, ytr)   # refit on train only
```
- Lab 6 used `cv=2` random folds + accuracy + `refit=True`: random folds mix past and future games; accuracy is meaningless at 15% positives; `refit=True` with a predefined split retrains on train + validation and we lose the validation score.

### Grid and result
`C` ∈ {0.01 … 3}, `l1_ratio` ∈ {0 (L2/Ridge), 1 (L1/Lasso)} with solver `saga` (`penalty` is deprecated in scikit-learn 1.9), `class_weight` ∈ {None, balanced}.
At **every T**: **L1**, **C = 0.1–0.3** (stronger regularisation than the baseline's C = 1), **no class weight**. PR-AUC 0.456 / 0.821 / 0.862 / 0.885; gain at T = 30 +0.052 [0.032, 0.074].
- L1 = built-in feature selection (many coefficients at 0) → consistent with notebook 06 (L1 selection keeps 33 features at T = 30).
- No class weight: re-weighting moves probabilities (helps the 0.5 threshold) but does not change the ranking much, and PR-AUC only measures the ranking.
- `logreg_C.png`: small C → bias (train and validation low); large C → train keeps rising, validation stalls (variance).

---

## 3. SVM, bagging and ensembles (`05_models.ipynb`, section 5)

### SVM (lab 6, ex. 2)
Finds the boundary with the **largest margin** between successes and failures; the RBF kernel allows a curved boundary. Grid `C` ∈ {0.1, 1, 10} (margin softness), `gamma` ∈ {scale, auto} (kernel width), `class_weight="balanced"`. Linear kernel not tested (another linear model; libsvm very slow on 7,000 games). The search scores `decision_function`; only the final model uses `probability=True` (Platt scaling, 5 internal folds, slow; deprecated in 1.9 → `CalibratedClassifierCV(SVC(), ensemble=False)` to adopt).
Result: 0.428 / 0.810 / 0.846 / 0.873 – below the tuned logistic regression, slowest fit (13–38 s per T).

### Bagging (lab 6, ex. 3)
Train copies of a model on random subsets of the games (`max_samples` 0.3/0.5, `n_estimators` 10/25), average them. Reduces **variance**.
- **Tree**: 0.901 → **0.916** at T = 30 (a deep tree has high variance) – the level of the best models.
- **Logistic regression**: 0.862 → 0.862 (low variance, nothing to gain).
- **SVM**: 0.846 → **0.769**, significantly worse than the baseline. **Flaw 9**: copies trained without probabilities, so the bagged probability is the share of SVMs voting "success" → only 11 distinct values for 1,499 games (verified in `val_proba.parquet`); ties break the ranking. Fix proposed (not run): SVMs with probabilities.
- **Hybrid soft voting** (bagged SVM + bagged tree): 0.904, below the bagged tree alone: averaging with a weaker model dilutes the better one.

### Voting and stacking of the best models (LR + RF + GB)
- **Soft voting**: average of the three probabilities → 0.914 at T = 30.
- **Stacking**: a logistic regression learns how to combine the three probabilities, trained on **out-of-fold** predictions (`KFold(5, shuffle=False)` on games sorted by date: each fold is a continuous period). Out-of-fold, because predictions on the games a model was trained on are over-confident. → 0.915 at T = 30, 0.946 at T = 90.
- Neither significantly beats gradient boosting alone (0.917): probably because the three models make similar errors, so averaging brings little.

---

## 4. Figures, hours leak, test (sections 7–9)
- `prauc_vs_T_models.png` (all models with CI), `confusion_T30.png`, comparison table with fit/predict times and interpretability, `val_proba.parquet` (validation probabilities, used by notebook 07).
- **Hours leak measured** (Flaw 2): adding `hours_mean`/`hours_median` back to the baseline at T = 30: 0.810 → 0.809, difference −0.001 [−0.005, +0.003].
- **Test**: `RUN_TEST = False`. In week 4, once: refit the chosen model on train + validation, evaluate on test with a bootstrap CI; the F1 threshold is chosen on validation.

### Comparison at T = 30 (to know)
| Model | PR-AUC | Fit (s) | Interpretability |
|---|---|---|---|
| Gradient boosting | 0.917 | 8.1 | low |
| Bagging of trees | 0.916 | 1.2 | low |
| Stacking | 0.915 | 49.3 | low |
| Random forest | 0.910 | 10.1 | low |
| Decision tree | 0.901 | 0.2 | high (rules) |
| LR tuned | 0.862 | 1.3 | high (coefficients) |
| SVM | 0.846 | 19.0 | very low |
| LR baseline | 0.810 | 0.3 | high |

### Possible questions
- *Why `refit=False`?* See the protocol above.
- *Why stacking with `shuffle=False`?* Shuffled folds would let the meta-model learn from predictions of models trained on future games.
- *Why does bagging help trees but not the logistic regression?* Bagging reduces variance; the logistic regression has little variance, a deep tree a lot.
- *Final model?* Gradient boosting: best or tied, 6× faster than stacking, simpler to explain and maintain.
