# Revision – Arthur DISERBEAU

Your parts: **Module 2 – target and split** (`02_target_split.ipynb`), **decision tree and random forest** (`05_models.ipynb`, section 2) and **uncertainty: bootstrap confidence intervals** (`05_models.ipynb`, section 6). Read `00_common.md` first.

---

## 1. Module 2 – target and split (`02_target_split.ipynb`)

### What the notebook does, step by step
1. **Checks Module 1 instead of trusting it:** recomputes `days` from the dates and asserts it equals the `days` column.
2. **Keeps only games with a full first year observed** (`release_date + 365 days <= last review`). Last review: 2022-12-31, games released up to 2021-12-31, so none is removed; the check protects future changes of the subset.
3. **`first_year_counts(games, reviews)`**: keeps reviews with `days < 365`, then per game:
   - `n_reviews_1y` = number of reviews (`size`);
   - `pos_ratio_1y` = mean of `is_recommended` (mean of True/False = share of positive reviews).
   - **Left join from all games**: a game with no review keeps `n_reviews_1y = 0` (filled) and `pos_ratio_1y` = NaN (unknown, not 0). Starting from the reviews would drop these 2,200 games (22%) and overestimate the success rate: **survivorship bias**.
4. **Toy test**: hand-made reviews at days 0, 10, 364 and 365 → asserts that day 0 and day 364 are counted and day 365 is not (`<` vs `<=`), and that a game without reviews is kept with an unknown ratio. Prints "Toy test passed".
5. **Time-based split on the dates**: `d_train` = 70th percentile of release dates, `d_val` = 85th. `train` if `release_date <= d_train`, `val` if `<= d_val`, else `test`. An assert checks that each split starts after the previous one ends: no release day is shared. Result: 7,007 / 1,499 / 1,494 games.
6. **Success**: `threshold` = 80th percentile of `n_reviews_1y` **on the train set only** = **66**; `success = (n_reviews_1y >= 66) & (pos_ratio_1y >= 0.80)`. A game without reviews has NaN ratio → comparison False → failure.
7. **Crosstab on train**: 1,407 games in the top 20%, of which 924 also have ≥ 80% positive (successes); 483 popular but too criticised.
8. **Leaky-threshold demonstration**: on all games the threshold would be 69 and 36 labels would change. Small here, but the label of a train game would depend on future games.
9. **Wilson 95% CI of the success rate per split**: train 13.2% [12.4; 14.0], val 15.7% [13.9; 17.6], test 14.9% [13.2; 16.8]. Wilson instead of `p ± 1.96·sqrt(p(1−p)/n)` because it stays in [0, 1] and is reliable for small proportions.
10. **Figures**: histogram of `log(1 + n_reviews_1y)` with the threshold, histogram of `pos_ratio_1y` (median 82%, 55.5% of games ≥ 80%, 15.9% at exactly 100%, 1.6% at 0%), success rate by release quarter (drops in 2017–2018, rises from 2019: distribution shift).

### Your AI audit entry – Flaw 6 (same-day games split)
The AI's temporary split cut on the **rank** of each game (`rank(method="first")`): 8 games released on 2020-05-20 were split 1 in train / 7 in validation, depending on row order. Fix: cut on the **dates** + assert. 13 games changed split; baseline at T = 0 0.411 → 0.407.

### Possible questions
- *Why top 20% AND 80% positive?* Number of reviews = popularity (linked to players); share of positive = satisfaction. A game played a lot but disliked is not a success. The 80% condition removes about a third of the popular games.
- *Why compute the threshold on train only?* Otherwise the label of training games depends on validation/test games: leakage.
- *Why Wilson?* See step 9. With p ≈ 0.14 and n ≈ 1,500 both give similar intervals here, but Wilson is the safe default.
- *The success rate changes over time – a problem?* Yes, a distribution shift: each model must be compared with the random level of the same split (0.157 on validation, not 0.132).

---

## 2. Decision tree (`05_models.ipynb`, section 2.1)

### Idea
A sequence of yes/no questions on the features (e.g. "is `n_reviews` above x?"). Each leaf predicts the share of successes of its training games. Readable as rules, but unstable (a small change in the data changes the tree): high **variance**.

### Grid search (full grid, 80 configurations)
`criterion` (gini / entropy: impurity measure used to choose splits), `max_depth` (3, 5, 8, 12, None), `min_samples_leaf` (1, 5, 20, 50), `class_weight` (None, balanced).
Best at T = 30: gini, depth 12, at least 50 games per leaf, no class weight → **PR-AUC 0.901** (0.402 / 0.837 / 0.901 / 0.925 over T). At T = 0 it does not beat the baseline (gain CI [−0.065, 0.055]).

### Bias / variance curve (Figure `tree_depth.png`)
With `min_samples_leaf = 1` and only the depth varying (T = 30):
- depth 1–3: train ≈ validation, both low (≈ 0.85 at depth 3) → **bias**;
- depth 5: best validation ≈ 0.87 (train ≈ 0.91);
- depth ≥ 12: train = 1 (every training game isolated), validation falls to ≈ 0.63 at unlimited depth → **variance** (gap ≈ 0.37).
The tuned tree (0.901) is better than the best depth-only tree because `min_samples_leaf = 50` also limits variance.

### Rules of a depth-3 tree (Figure `tree_T30.png`, values scaled)
- Root: `n_reviews` (log, scaled) ≤ 0.785 → 80.6% of games, only 2% successes.
- Otherwise (19.4% of games, 60% successes): split on `pos_ratio`; games with high `pos_ratio` **and** many reviews (`n_reviews` > 0.941) → 10.5% of games with **91.6% successes**.
- The tree uses only the two review signals: same message as the permutation importance (notebook 06).

## 3. Random forest (`05_models.ipynb`, section 2.2)
Many deep trees, each on a **bootstrap sample** of games (= bagging of trees) and a random subset of features at each split (`max_features`); averaging reduces the variance. Random search (8 configurations) on `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `class_weight`. **0.489 / 0.866 / 0.910 / 0.944**; at T = 0 it is the best single model (only voting 0.495 and stacking 0.496 do slightly better). `n_jobs=1` inside the search because the search itself is parallel (AI_AUDIT, Flaw 7: 16 × 16 threads saturated the CPU).

---

## 4. Uncertainty (`05_models.ipynb`, section 6)

### Code
```python
BOOT_IDX = [i for i in (rng.integers(0, len(y_val), len(y_val)) for _ in range(N_BOOT)) if y_val[i].any()]
BOOT = {key: boot_scores(p) for key, p in PROBA.items()}          # 1,000 PR-AUC per model and T
results["CI_low"] = 2.5th percentile, results["CI_high"] = 97.5th percentile
delta = BOOT[model] - BOOT[baseline]                               # paired: same resampled games
significant = 2.5th percentile of delta > 0
```
- The **same** resampled indices are used for every model → paired comparison (the common variability, e.g. an "easy" resample, cancels out).
- Resamples without any success are skipped (PR-AUC undefined).

### Results to know
- CI of the best model: [0.43, 0.56] at T = 0, [0.89, 0.94] at T = 30 → wide, because the validation set has only 235 successes.
- Gradient boosting vs baseline at T = 30: +0.107 [0.076, 0.142] → significant.
- At T = 30 the five best models are within 0.007 with overlapping intervals: **we cannot rank them**.
- Not significant: decision tree, bagged tree, SVM, MLP at T = 0. Bagged SVM significantly **worse** at T = 0, 7, 30.

### Possible questions
- *Assumptions of the bootstrap?* Games are independent and the validation set represents the population. Games of the same publisher or month are correlated → intervals rather optimistic.
- *Multiple comparisons?* 13 models compared on one validation set: the best score is slightly optimistic (winner's curse). The single test evaluation corrects this.
- *Why paired?* Two models scored on the same games are correlated; comparing two separate intervals would be too conservative. The CI of the difference is the right test.
