# Répartition des modules – Projet ML Steam (tout le projet en une journée)

> **Rappel des règles (Project Brief & Governance Policy)**
> - 4 modules, **1 responsable par module**, à reporter dans `ai_governance/CONTRIBUTIONS.md`.
> - Chacun écrit **au moins 1 entrée** dans `PROMPT_LOG.md` **et** au moins 1 dans `AI_AUDIT.md` (sinon −2 pts individuels).
> - À l'oral (3 min par personne), chacun explique **chaque ligne de son module** et dit ce qui vient de lui et ce qui vient de l'IA. Il y a aussi des questions de rigueur statistique sur **tout** le projet.
> - Scaler, imputer ou SMOTE ajustés avant le découpage train/val = **note technique plafonnée à 10/20**.
> - Rapport **en anglais** (sinon note divisée par 2), en LaTeX (bonus +1/+2).

Les prompts sont écrits en anglais pour pouvoir être copiés tels quels dans `PROMPT_LOG.md`. Ce sont des **exemples à adapter** : notez la version que vous avez vraiment utilisée.

---

## 0. Périmètre réduit (tout est à rendre aujourd'hui)

- **SteamCharts : abandonné.**
- **Sous-ensemble de jeux** pour aller vite, par exemple les jeux sortis entre 2015 et 2021, ou un échantillon de quelques milliers d'`app_id`.
  ⚠️ **On échantillonne par jeu** (on garde tous les avis des jeux tirés), **jamais par ligne d'avis**. Sinon le nombre d'avis par jeu, donc la cible, est faussé.
- **Données utilisées :**
  - Kozyriev : `recommendations.csv` et `games.csv` (pas `users.csv`). `games_metadata.json` (tags, description) est optionnel.
  - FronkonGames : `games.csv` avec `usecols`, après avoir vérifié que les colonnes ne sont pas décalées.
- **Modèles :** baselines (`DummyClassifier`, régression logistique) → random forest, `HistGradientBoostingClassifier` → `MLPClassifier` (modèle hors cours, avec un article de référence) → ensemble par vote.
- **Règle d'or :** une version qui tourne de bout en bout (baseline et tableau de résultats) **avant** de passer aux modèles avancés.

---

## 1. Conventions communes (créneau 0, tous ensemble, 30 min)

**Structure du dépôt (= structure du .zip)**
```
Group_XX_DS_Project/
├── requirements.txt          # versions fixées (pip freeze)
├── config.py                 # chemins, SEED, constantes
├── notebooks/
│   ├── 01_data.ipynb         # Module 1
│   ├── 02_target_split.ipynb # Module 2
│   ├── 03_features.ipynb     # Module 3
│   └── 04_models.ipynb       # Module 4
├── data/raw/  data/processed/   # pas dans le .zip si trop lourd (expliquer comment les télécharger)
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
HORIZON_DAYS = 365          # succès mesuré à 1 an
T_VALUES = [0, 7, 30, 90]   # jours observés après la sortie
TOP_QUANTILE = 0.80         # top 20 % en nombre d'avis (calculé sur le TRAIN)
MIN_POS_RATIO = 0.80        # au moins 80 % d'avis positifs
SPLIT = (0.70, 0.15, 0.15)  # train / val / test, par date de sortie
RAW_DIR, PROC_DIR = "data/raw", "data/processed"
```

**Règles**
- Clé commune : **`app_id`** (entier). Après chaque jointure : `assert df["app_id"].is_unique`.
- Pendant le développement, tout le monde travaille sur un **échantillon** (`nrows=...`). Le sous-ensemble final ne sert qu'à partir du créneau 2.
- Colonnes **interdites comme variables** (mesurées après la sortie) : `positive`, `negative`, `positive_ratio`, `user_reviews`, `rating`, `recommendations`, `estimated_owners`, `peak_ccu`, temps de jeu, `user_score`, `metacritic_score`, `price_final`, `discount`, `steam_deck`.
- **Le test n'est utilisé qu'une seule fois**, au créneau 4. Tous les choix se font sur la validation.

**Fichiers produits par chaque module**

| Module | Fichier | Colonnes minimum |
|---|---|---|
| 1 | `games_clean.parquet` | `app_id`, `release_date`, colonnes nettoyées (prix de base, plateformes, langues, genres, tags…) |
| 1 | `reviews_clean.parquet` | `app_id`, `date`, `days` (jours depuis la sortie), `is_recommended`, `hours`, `helpful` |
| 2 | `target.parquet` | `app_id`, `release_date`, `n_reviews_1y`, `pos_ratio_1y`, `success`, `split` |
| 3 | `features_static.parquet` | `app_id` + variables connues à la sortie |
| 3 | `features_T{7,30,90}.parquet` | `app_id` + signaux précoces |
| 4 | `results.csv` | modèle, T, PR-AUC, ROC-AUC, F1, IC 95 % |

**Correspondance avec les objectifs de la semaine 2 (à recopier dans `CONTRIBUTIONS.md`)**

| Objectif | Module(s) |
|---|---|
| Clean and preprocess the dataset | 1 (nettoyage brut), 3 (transformation en variables), 4 (imputation et normalisation dans le pipeline) |
| Analyze features, missing values and target distribution | 3 (variables, valeurs manquantes), 2 (cible) |
| Define training, validation and test sets | 2 |
| Select and implement a baseline model | 4 |
| Evaluate initial performance using appropriate metrics | 4 (métriques), 2 (intervalles de confiance) |

---

## 2. Planning de la journée

| Créneau | Module 1 – Données | Module 2 – Cible et découpage | Module 3 – Variables | Module 4 – Modèles |
|---|---|---|---|---|
| **0. Ensemble (30 min)** | Dépôt, `config.py`, `requirements.txt`, colonnes des fichiers, dossier `ai_governance/` | ← | ← | ← |
| **1. Préparation (~1 h 30)** | Nettoyage et fusion, **version 0 sur échantillon en moins d'une heure**, puis calcul de `days` | Code de la cible sur l'échantillon brut | Code des variables statiques et des signaux T sur l'échantillon | Pipeline et baselines sur un petit tableau, **squelette du rapport LaTeX** |
| **2. Baseline (~1 h 30)** | Lancement sur le sous-ensemble final, `assert` de vérification | Découpage par date, seuil sur le train, distribution de la cible | Valeurs manquantes, corrélations, liste des colonnes exclues | Assemblage, **baselines et premier tableau de résultats** |
| **3. Modèles (~2 h)** | `MLPClassifier` + article scientifique, ensemble par vote | Déséquilibre des classes, intervalles de confiance, comparaison des modèles | Sélection de variables, importance des variables pour chaque T | Random forest, gradient boosting, réglage, **courbe en fonction de T** |
| **4. Finalisation (~1 h 30)** | Analyse des erreurs (types de jeux mal prédits) | Partie « incertitude » | Interprétation : qu'est-ce qui fait un succès ? | **Évaluation finale sur le test, une seule fois** |
| **Tout le monde, en fin de journée** | Sa section du rapport (en anglais), ses entrées `PROMPT_LOG` et `AI_AUDIT`, relecture du module suivant (1→2→3→4→1) | ← | ← | ← + assemblage du .zip |

**Si on manque de temps :** au créneau 3, on garde seulement la régression logistique et le gradient boosting, et on supprime l'ensemble de modèles. Le MLP reste, car un algorithme hors cours est demandé.

---

## Module 1 – Données : chargement, nettoyage, fusion, modèle hors cours

**Responsable :** ______________

**Ce qui est dans le module**
- *Créneaux 1-2 :*
  - Charger Kozyriev (`games.csv`, `recommendations.csv`) et FronkonGames (`games.csv` avec `usecols`), avec des types compacts.
  - Vérifier l'alignement des colonnes du CSV FronkonGames, et redécouper les listes stockées en texte (tags, genres, langues).
  - Convertir les dates, supprimer les doublons, retirer les valeurs impossibles.
  - Fusionner les deux tables de jeux sur `app_id` (base = Kozyriev). Compter les jeux perdus, et comparer les deux dates de sortie.
  - Rattacher `release_date` aux avis et calculer `days`. Compter les avis avec `days < 0` (accès anticipé) et les exclure.
  - Tirer le sous-ensemble de jeux, **par `app_id`**.
- *Créneau 3 :* `MLPClassifier` comme modèle hors cours, justifié par un article scientifique ; ensemble des meilleurs modèles par vote.
- *Créneau 4 :* analyse des erreurs. Sur quels jeux le modèle se trompe-t-il (genre, prix, nombre d'avis) ?

**Idées de prompts**

1. *Chargement et inspection*
   > I have `recommendations.csv` (41M rows; columns: app_id int, helpful int, funny int, date str YYYY-MM-DD, is_recommended bool, hours float, user_id int, review_id int) and `games.csv` (app_id, title, date_release, win, mac, linux, rating, positive_ratio, user_reviews, price_final, price_original, discount, steam_deck). Write pandas code that loads only app_id, date, is_recommended, hours, helpful from recommendations with compact dtypes (int32, bool, float32), parses dates, and prints shape, dtypes, % missing per column and the number of duplicated rows. Do not drop or impute anything yet. Tell me the expected memory usage and how to read the file in chunks if I have less than 12 GB of RAM.

2. *Fusion et sous-ensemble*
   > Merge the Kozyriev games table (base table) with the FronkonGames table (key `AppID` renamed to `app_id`) using a left join with `validate="one_to_one"` and `indicator=True`. Report the number of games in each source, how many are kept, and compare the two release dates (flag differences > 30 days). Then draw a random sample of N app_id (seed fixed) and keep ALL reviews of these games: do not sample individual review rows. Explain why sampling review rows would bias the number of reviews per game.

3. *Modèle hors cours*
   > Add an MLPClassifier to the existing scikit-learn pipeline (imputer + scaler fitted on train only). Use early_stopping=True, random_state=SEED, and tune hidden_layer_sizes and alpha on the validation set only (PredefinedSplit). Compare it with the logistic regression baseline on PR-AUC. Give me one peer-reviewed reference that justifies using a neural network on tabular data, and tell me its main limitation compared to gradient boosting.

**Erreurs de l'IA à surveiller**
- Une fusion qui **duplique** des lignes (`app_id` pas unique).
- Des dates mal lues (« Coming soon », format américain).
- Un échantillonnage **par ligne d'avis** au lieu de par jeu.
- Le fichier d'avis chargé en entier avec les types par défaut, ce qui fait planter le PC.
- Une **référence scientifique inventée** : vérifiez que l'article existe vraiment, sur Google Scholar.

**Questions possibles à l'oral :** pourquoi une jointure `left` ? Combien de jeux perdus, et est-ce que ça biaise l'échantillon ? Pourquoi échantillonner par jeu ? Pourquoi le MLP fait-il mieux ou moins bien que le gradient boosting ?

---

## Module 2 – Cible, découpage et incertitude

**Responsable :** ______________

**Ce qui est dans le module**
- *Créneaux 1-2 :*
  - Pour chaque jeu, `n_reviews_1y` et `pos_ratio_1y` sur les avis avec `0 ≤ days < 365`. On part de **tous les jeux** (jointure `left`) : un jeu sans avis a 0 avis, ce n'est pas une ligne à supprimer.
  - Garder les jeux qui ont un an complet d'avis.
  - Découpage par date de sortie (70 / 15 / 15), puis seuil du top 20 % calculé **sur le train**, et enfin `success`.
  - Distribution de la cible par split et dans le temps.
- *Créneau 3 :*
  - Déséquilibre des classes : `class_weight` comparé à un rééchantillonnage **dans** le pipeline.
  - Intervalles de confiance à 95 % par bootstrap sur la validation.
  - Comparaison des modèles : la différence est-elle significative ?
- *Créneau 4 :* partie « incertitude et limites » du rapport.

**Idées de prompts**

1. *Calcul de la cible*
   > Using `reviews_clean` (app_id, days, is_recommended) and `games_clean` (app_id, release_date), compute for each game `n_reviews_1y` and `pos_ratio_1y` using only reviews with `0 <= days < 365`. Start from ALL games in games_clean with a left join, so that games with zero reviews get n_reviews_1y = 0 instead of disappearing. Keep only games where `release_date + 365 days <= max(review date)`. Use vectorized pandas and add an assert that no review outside the window was used.

2. *Découpage et étiquette*
   > Split the games by release_date: oldest 70% = train, next 15% = validation, most recent 15% = test (no shuffling). Compute the 80th percentile of `n_reviews_1y` on the TRAIN set only and use it for all splits: `success = (n_reviews_1y >= threshold) & (pos_ratio_1y >= 0.8)`. Print the threshold, the date range and the positive rate of each split. Explain why computing the threshold on the full dataset would be data leakage.

3. *Incertitude et comparaison*
   > Given validation labels and the predicted probabilities of two models, compute a 95% bootstrap confidence interval (1000 resamples, fixed seed) for the PR-AUC of each model, and a paired bootstrap test on the PR-AUC difference (resample the same indices for both models). State the assumptions of this test and when it would not be valid.

**Erreurs de l'IA à surveiller**
- Un **`train_test_split` aléatoire** au lieu d'un découpage par date.
- Un seuil calculé sur **toutes** les données.
- Des jeux sans avis **supprimés** au lieu d'être gardés avec 0 avis : biais du survivant.
- SMOTE ou rééchantillonnage appliqué **avant** le découpage : note plafonnée à 10/20.
- Un bootstrap non apparié pour comparer deux modèles.

**Questions possibles à l'oral :** pourquoi un découpage temporel ? Que se passe-t-il en cas de changement de distribution entre jeux anciens et récents ? Pourquoi « top 20 % » et pas un seuil fixe ? Que veut dire un intervalle de confiance qui chevauche celui de la baseline ?

---

## Module 3 – Variables, analyse exploratoire et sélection

**Responsable :** ______________

**Ce qui est dans le module**
- *Créneaux 1-2 :*
  - **Variables statiques** : log du prix de base, gratuit ou non, plateformes, nombre de langues, mois et jour de sortie, longueur de la description, tags, genres et catégories en variables binaires (tags gardés selon leur fréquence **sur le train**).
  - **Signaux précoces** pour T = 7, 30 et 90 : nombre d'avis, % positifs, heures de jeu moyennes et médianes, votes « utile ». Les valeurs restent vides s'il n'y a pas d'avis, avec un indicateur `has_reviews`.
  - Analyse exploratoire : valeurs manquantes par split, distributions, valeurs aberrantes, corrélations.
  - **Audit des fuites de données** : liste des colonnes exclues, avec la raison.
- *Créneau 3 :* sélection de variables **dans le pipeline** (filtre, L1, RFE, comme au TP 4) ; importance des variables pour chaque T (importance par permutation sur la validation).
- *Créneau 4 :* interprétation. Quelles variables comptent à T = 0, puis à T = 30 ?

**Idées de prompts**

1. *Variables statiques*
   > From these columns [price_original, win, mac, linux, Supported languages, release_date, About the game, Genres, Categories, Tags, DLC count, Required age], build features known at release time: log1p(price), is_free, platform flags, n_languages, release month and weekday, description length, and multi-hot encoding of genres/categories/tags keeping only tags present in at least 1% of TRAIN games. For each column, tell me whether it could have changed after release (e.g. DLC count, tags) and flag it as a leakage risk.

2. *Signaux précoces*
   > For each T in [7, 30, 90], compute per game from `reviews_clean` with `0 <= days < T`: n_reviews, pos_ratio, mean and median hours, total helpful votes. Start from all games (left join): n_reviews = 0 when there is no review, pos_ratio and hours stay NaN, and add a has_reviews flag. Do not impute anything here (imputation is done in the model pipeline). Add an assert that no review with days >= T is used.

3. *Sélection et importance*
   > Add feature selection INSIDE the scikit-learn pipeline (SelectKBest with mutual_info_classif, and SelectFromModel with an L1 logistic regression), tune k or C on the validation set only, and report which features are kept for T = 0 and T = 30. Then compute permutation importance on the validation set for the best model. Explain why selecting features on the full dataset before splitting would be leakage.

**Erreurs de l'IA à surveiller**
- L'IA propose `positive_ratio`, `user_reviews`, `estimated_owners` ou `peak_ccu` comme variables : **fuite de données**.
- Une fenêtre qui utilise des avis **après T**.
- Les tags gardés selon leur fréquence sur **toutes** les données.
- Des valeurs manquantes remplacées par 0 sans le dire (0 avis ≠ donnée manquante).
- Une sélection de variables faite **avant** le découpage.

**Questions possibles à l'oral :** pourquoi exclure telle colonne ? Que se passe-t-il si un tag a été ajouté après la sortie ? Le prix est corrélé au succès : est-ce une relation de cause ?

---

## Module 4 – Pipeline, modèles et évaluation finale

**Responsable :** ______________

**Ce qui est dans le module**
- *Créneaux 1-2 :*
  - Assembler les fichiers des modules 2 et 3 sur `app_id` : un jeu de données par valeur de T.
  - Pipeline scikit-learn : imputation, normalisation et encodage **dans** le pipeline, ajustés sur le train seulement.
  - Baselines `DummyClassifier` et régression logistique (T = 0).
  - Métriques sur la validation : **PR-AUC (principale)**, ROC-AUC, F1, précision, rappel, matrice de confusion. Comparer avec la PR-AUC d'un modèle au hasard (= proportion de succès).
  - Squelette du rapport LaTeX.
- *Créneau 3 :* random forest, `HistGradientBoostingClassifier`, réglage avec `RandomizedSearchCV` + `PredefinedSplit` (validation par date) ; **courbe « PR-AUC en fonction de T »**.
- *Créneau 4 :* évaluation finale **une seule fois** sur le test, comparaison avec la baseline, assemblage du .zip.

**Idées de prompts**

1. *Pipeline et baseline*
   > Build a scikit-learn Pipeline with a ColumnTransformer: numerical features → SimpleImputer(strategy="median") then StandardScaler; binary multi-hot features → passthrough. The pipeline must be fitted on TRAIN only. Models: DummyClassifier(strategy="most_frequent") and LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED). Evaluate on VALIDATION with average_precision_score (PR-AUC), roc_auc_score, F1, precision, recall and the confusion matrix. Also print the PR-AUC of a random classifier (= positive rate). Do not use the test set.

2. *Réglage et courbe en fonction de T*
   > Tune a HistGradientBoostingClassifier and a RandomForestClassifier with RandomizedSearchCV (20 iterations, random_state=SEED) using a PredefinedSplit where the fold is our time-based validation set (no random CV, because our split is temporal). Use average_precision as the scoring. Repeat for T in [0, 7, 30, 90] with the same target and split, and plot validation PR-AUC versus T for each model, with the baseline as a horizontal line.

3. *Évaluation finale*
   > Refit the selected model (chosen on validation only) on train+validation, then evaluate it ONCE on the test set: PR-AUC, ROC-AUC, F1 at the threshold chosen on validation, with 95% bootstrap confidence intervals. Explain why the test score can be lower than the validation score.

**Erreurs de l'IA à surveiller**
- `StandardScaler` ou `SimpleImputer` ajustés **avant** le découpage : **note plafonnée à 10/20**.
- **L'accuracy** comme métrique principale avec des classes déséquilibrées.
- `roc_auc_score` calculé sur les classes prédites au lieu de `predict_proba`.
- Une validation croisée **aléatoire** pour le réglage, alors que le découpage est temporel.
- Le test utilisé pour choisir le modèle ou le seuil.
- `random_state` oublié : résultats différents à chaque exécution.

**Questions possibles à l'oral :** pourquoi la PR-AUC plutôt que l'accuracy ? Que devient la PR-AUC si la proportion de succès change ? Que fait `class_weight="balanced"` ? Pourquoi un `PredefinedSplit` ?

---

## Rapport (en anglais, LaTeX)

Chacun rédige la section de son module. Le module 4 assemble.

| Section du rapport | Qui |
|---|---|
| Business scope, problem formalisation | Tous (repris de la spécification) |
| Data description, cleaning, merging, sampling | Module 1 |
| Target definition, data splitting, class imbalance | Module 2 |
| Features, missing values, outliers, leakage audit, feature selection | Module 3 |
| Models, hyperparameters, results vs baseline, performance vs T | Module 4 |
| Out-of-course model (MLP) + scientific reference, error analysis | Module 1 |
| Uncertainty (confidence intervals), limitations | Module 2 |
| Discussion and conclusion | Tous |

---

## Modèles d'entrée pour les fichiers de gouvernance

**`PROMPT_LOG.md`**
```markdown
### Entry N – <titre> (Module X – <prénom>)
- **Goal:** ...
- **Bare prompt (v1):** "..."  → problem with the answer: ...
- **Constrained prompt (v2):** "..." (schema, leakage boundaries, assumptions, fallback)
- **What we kept / changed:** ...
```

**`AI_AUDIT.md`**
```markdown
### Flaw N – <titre> (Module X – <prénom>)
- **AI output:** what the AI generated (code snippet)
- **Problem:** what is wrong + root cause (e.g. scaler fitted on full data → leakage)
- **How we detected it:** test, assert, metric that looked too good...
- **Fix:** corrected code
- **Impact:** metric before / after the fix
```
