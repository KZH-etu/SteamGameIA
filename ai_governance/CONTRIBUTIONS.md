# Contributions

| Module | Title | Owner |
|---|---|---|
| 1 | Data: loading, cleaning, merging | Yohan ALLOUCHE |
| 2 | Target and train / validation / test split | Arthur DISERBEAU |
| 3 | Features and exploratory analysis | Jason ZOH |
| 4 | Baseline and evaluation | Lucas GREZES |

## Mapping with the week 2 objectives

| Objective | Module(s) |
|---|---|
| Clean and preprocess the dataset | 1 (raw cleaning), 3 (transformation into features), 4 (imputation and scaling in the pipeline) |
| Analyze features, missing values and target distribution | 3 (features, missing values), 2 (target) |
| Define training, validation and test sets | 2 |
| Select and implement a baseline model | 4 |
| Evaluate initial performance using appropriate metrics | 4 (metrics), 2 (confidence intervals) |

## Week 3 – Model development and experiments

| Part | Notebook | Owner |
|---|---|---|
| Grid search on the baseline logistic regression | `05_models.ipynb`, section 1 | Lucas GREZES |
| Decision tree (bias / variance, rules) and random forest | `05_models.ipynb`, section 2 | Arthur DISERBEAU |
| Gradient boosting | `05_models.ipynb`, section 3 | Yohan ALLOUCHE |
| MLP (out-of-course model) | `05_models.ipynb`, section 4 | Jason ZOH |
| SVM, bagging (SVM, tree, logistic regression), Voting and Stacking | `05_models.ipynb`, section 5 | Lucas GREZES |
| Uncertainty: bootstrap confidence intervals and paired comparison with the baseline | `05_models.ipynb`, section 6 | Arthur DISERBEAU |
| Figures, comparison table, hours leak, final test evaluation | `05_models.ipynb`, sections 7–9 | Lucas GREZES |
| Feature selection (filter, embedded, wrapper) and permutation importance | `06_feature_selection.ipynb` | Jason ZOH |
| Error analysis | `07_error_analysis.ipynb` | Yohan ALLOUCHE |
| Integration: English translation, sections per owner, `main.py` (01 → 07) | all notebooks | Jason ZOH |

Each owner checks the code of their section and writes its "Reading (…, after the run)" paragraph.

## AI governance per member

| Member | PROMPT_LOG | AI_AUDIT |
|---|---|---|
| Yohan ALLOUCHE | Entries 1, 2, 9 | Flaws 1, 8 (gradient boosting grid) |
| Arthur DISERBEAU | Entries 5, 7 (decision tree) | Flaw 6 |
| Jason ZOH | Entries 3, 8 | Flaws 2, 3, 5, 8 (MLP grid) |
| Lucas GREZES | Entries 4, 6, 7 (SVM, bagging) | Flaws 4, 7, 9 |
