# Répartition des modules – Projet ML Steam

> **Rappel des règles (Project Brief & Governance Policy)**
> - 4 modules, **1 responsable par module**, à reporter dans `ai_governance/CONTRIBUTIONS.md`.
> - Chacun écrit **au moins 1 entrée** dans `PROMPT_LOG.md` **et** au moins 1 dans `AI_AUDIT.md` (sinon −2 pts individuels).
> - À l'oral (3 min par personne), chacun doit expliquer **chaque ligne de son module** et dire ce qui vient de lui et ce qui vient de l'IA.
> - Scaler, imputer ou SMOTE ajustés avant le découpage train/val = **note technique plafonnée à 10/20**.

Les prompts sont écrits en anglais pour pouvoir être copiés tels quels dans `PROMPT_LOG.md`. Ce sont des **exemples à adapter** : notez la version que vous avez vraiment utilisée.

---

## 0. Conventions communes (à fixer ensemble au début)

**Structure du dépôt**
```
Group_XX_DS_Project/
├── requirements.txt          # versions fixées (pip freeze)
├── config.py                 # chemins, SEED, constantes
├── notebooks/
│   ├── 01_data.ipynb         # Module 1
│   ├── 02_target_split.ipynb # Module 2
│   ├── 03_features_eda.ipynb # Module 3
│   └── 04_baseline.ipynb     # Module 4
├── data/raw/  data/processed/   # pas sur Moodle si trop lourd
├── report/
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
TOP_QUANTILE = 0.80         # top 20 % en nombre d'avis
MIN_POS_RATIO = 0.80        # au moins 80 % d'avis positifs
RAW_DIR, PROC_DIR = "data/raw", "data/processed"
```

**Règles**
- Clé commune : **`app_id`** (entier).
- Pendant le développement, travaillez sur un **échantillon** (`nrows=...` ou quelques milliers de jeux). Le fichier complet ne sert qu'à la fin.
- Colonnes **interdites comme variables** (mesurées après la sortie) : `positive`, `negative`, `positive_ratio`, `user_reviews`, `rating`, `recommendations`, `estimated_owners`, `peak_ccu`, temps de jeu, `user_score`, `metacritic_score`, `price_final`, `discount`.

**Fichiers produits par chaque module**

| Module | Fichier | Colonnes minimum |
|---|---|---|
| 1 | `games_clean.parquet` | `app_id`, `release_date`, colonnes nettoyées de FronkonGames |
| 1 | `reviews_clean.parquet` | `app_id`, `date`, `is_recommended`, `hours`, `helpful` |
| 2 | `target.parquet` | `app_id`, `release_date`, `n_reviews_1y`, `pos_ratio_1y`, `success`, `split` |
| 3 | `features_static.parquet` | `app_id` + variables connues à la sortie |
| 3 | `features_T{0,7,30,90}.parquet` | `app_id` + signaux précoces |
| 4 | `results_baseline.csv` | modèle, T, PR-AUC, ROC-AUC, F1, IC 95 % |

---

## Module 1 – Données : chargement, nettoyage, fusion

**Responsable :** ______________

**Ce qui est dans le module**
- Charger les deux sources : FronkonGames (fiche des jeux) et Kozyriev (`games.csv`, `recommendations.csv`, `games_metadata.json`).
- Lire seulement les colonnes utiles, avec des types compacts (le fichier d'avis est très gros).
- Convertir les dates, nettoyer les types, supprimer les doublons, gérer les jeux sans date ou avec une date invalide.
- Fusionner sur `app_id` et documenter combien de jeux sont perdus à chaque étape.
- Vérifier que les dates de sortie des deux sources sont cohérentes.
- *Plus tard (optionnel) :* scraping SteamCharts.

**Idées de prompts**

1. *Chargement et inspection*
   > I have `recommendations.csv` (41M rows; columns: app_id int, helpful int, funny int, date str YYYY-MM-DD, is_recommended bool, hours float, user_id int, review_id int) and `games.csv` (app_id, title, date_release, win, mac, linux, rating, positive_ratio, user_reviews, price_final, price_original, discount, steam_deck). Write pandas code that loads only app_id, date, is_recommended, hours, helpful from recommendations with compact dtypes (int32, bool, float32), parses dates, and prints shape, dtypes, % missing per column and the number of duplicated rows. Do not drop or impute anything yet. Tell me the expected memory usage and how to read the file in chunks if I have less than 12 GB of RAM.

2. *Fusion*
   > Merge the cleaned Kozyriev games table with the FronkonGames table (key: `AppID` renamed to `app_id`). Use `validate="one_to_one"` and report: number of games in each source, number kept after the merge, and the reasons for the losses. Compare the two release dates and flag games where they differ by more than 30 days. Do not create any feature and do not use columns measured after release (positive_ratio, user_reviews, estimated_owners, peak_ccu…).

**Erreurs de l'IA à surveiller (idées pour AI_AUDIT)**
- Une fusion qui **duplique** des lignes, parce que `app_id` n'est pas unique dans une des sources.
- Des dates mal lues : format américain, ou texte comme « Coming soon ».
- Le fichier d'avis chargé en entier avec les types par défaut, ce qui fait planter le PC.
- Des lignes supprimées sans le dire (`dropna()` sur tout le tableau).

**Questions possibles à l'oral :** pourquoi cette jointure (inner ou left) ? Combien de jeux perdus, et ça biaise-t-il l'échantillon ? Que se passe-t-il si un `app_id` est en double ?

---

## Module 2 – Cible et découpage train / validation / test

**Responsable :** ______________

**Ce qui est dans le module**
- Pour chaque jeu : nombre d'avis et % d'avis positifs **dans les 365 jours après la sortie**.
- Garder seulement les jeux qui ont au moins un an de recul (`release_date + 365 jours ≤ date du dernier avis du dataset`).
- Découpage **par date de sortie** : 70 % les plus anciens pour le train, 15 % pour la validation, 15 % les plus récents pour le test.
- Seuil « top 20 % » calculé **sur le train uniquement**, puis appliqué partout. Ensuite : `success = n_reviews_1y ≥ seuil ET pos_ratio_1y ≥ 0,8`.
- Analyse de la cible : proportion de succès par split, évolution dans le temps.
- *Plus tard :* gestion du déséquilibre (`class_weight`), intervalles de confiance.

**Idées de prompts**

1. *Calcul de la cible*
   > Using `reviews_clean` (app_id, date, is_recommended) and `games_clean` (app_id, release_date), compute for each game `n_reviews_1y` and `pos_ratio_1y` using only reviews with `release_date <= date < release_date + 365 days`. Count separately the reviews posted before release_date and do not use them. Keep only games where `release_date + 365 days <= max(date)` in the dataset. Use vectorized pandas (no Python loops) and add an assert that checks no review outside the window was used.

2. *Découpage et étiquette*
   > Split the games by release_date: oldest 70% = train, next 15% = validation, most recent 15% = test (no shuffling). Then compute the 80th percentile of `n_reviews_1y` on the TRAIN set only and use this threshold for all splits: `success = (n_reviews_1y >= threshold) & (pos_ratio_1y >= 0.8)`. Print the threshold, the date range and the positive rate of each split. Explain why computing the threshold on the full dataset would be data leakage.

**Erreurs de l'IA à surveiller**
- Un **`train_test_split` aléatoire** au lieu d'un découpage par date.
- Un seuil calculé sur **toutes** les données au lieu du train seulement.
- Une fenêtre qui inclut les avis **avant** la sortie, ou qui dépasse 365 jours (`<=` au lieu de `<`).
- Des jeux récents gardés alors qu'ils n'ont pas encore un an d'avis : leur cible serait fausse.

**Questions possibles à l'oral :** pourquoi un découpage temporel ? Que se passe-t-il si les jeux récents ressemblent moins aux anciens (changement de distribution) ? Pourquoi « top 20 % » et pas un seuil fixe ?

---

## Module 3 – Variables et analyse exploratoire

**Responsable :** ______________

**Ce qui est dans le module**
- **Variables connues à la sortie** (FronkonGames) : prix de lancement (en log), gratuit ou non, plateformes, nombre de langues, mois et jour de sortie, longueur de la description, tags, genres et catégories en variables binaires.
- **Signaux précoces** pour chaque T (Kozyriev) : nombre d'avis, % positifs, heures de jeu moyennes et médianes, votes « utile », tendance des avis.
- Analyse exploratoire : valeurs manquantes, distributions, valeurs aberrantes, corrélations, lien entre les variables et la cible.
- **Audit des fuites de données** : liste des colonnes exclues, avec la raison.
- *Plus tard :* sélection de variables (filtre, L1, RFE, comme au TP 4).

**Idées de prompts**

1. *Variables connues à la sortie*
   > From the FronkonGames columns [Price, Windows, Mac, Linux, Supported languages, Release date, About the game, Genres, Categories, Tags, DLC count, Required age], build features known at release time: log1p(price), is_free, platform flags, n_languages, release month and weekday, description length, and multi-hot encoding of genres/categories/tags keeping only tags present in at least 1% of TRAIN games. For each column, tell me whether it could have changed after release (e.g. DLC count, tags, current price) and flag it as a leakage risk.

2. *Signaux précoces*
   > For each T in [0, 7, 30, 90], compute per game from reviews with `release_date <= date < release_date + T days`: n_reviews_T, pos_ratio_T (NaN if no review, plus a has_reviews_T flag), mean and median hours, total helpful votes, and the number of reviews in the last 7 days of the window. For T = 0, all early-signal features must be empty. Add an assert that no review with date >= release_date + T is used.

3. *Analyse exploratoire*
   > Show the % of missing values per feature, separately for train/val/test, and test whether missingness is related to the target. Plot the distribution of the numerical features (log scale if skewed) and the correlation matrix. List pairs with |correlation| > 0.9. Use the train set only for any decision.

**Erreurs de l'IA à surveiller**
- L'IA propose `positive_ratio`, `user_reviews`, `estimated_owners` ou `peak_ccu` comme variables : **fuite de données**.
- Une fenêtre de signaux précoces qui utilise des avis **après T**.
- Les tags gardés selon leur fréquence sur **toutes** les données au lieu du train.
- Un prix actuel utilisé à la place du prix de lancement.
- Des valeurs manquantes remplacées par 0 sans le dire (0 avis n'est pas la même chose qu'une donnée manquante).

**Questions possibles à l'oral :** pourquoi exclure telle colonne ? Que se passe-t-il si un tag a été ajouté après la sortie ? Corrélation ou causalité entre le prix et le succès ?

---

## Module 4 – Baseline et évaluation

**Responsable :** ______________

**Ce qui est dans le module**
- Pipeline scikit-learn : imputation, normalisation et encodage **dans** le pipeline, ajustés sur le train seulement.
- Baseline 0 : `DummyClassifier` (classe majoritaire). Baseline 1 : régression logistique à T = 0.
- Évaluation sur la validation : **PR-AUC (métrique principale)**, ROC-AUC, F1, précision, rappel, matrice de confusion.
- Intervalle de confiance à 95 % par bootstrap, et comparaison avec la PR-AUC d'un modèle au hasard (= proportion de succès).
- Tableau des premiers résultats pour chaque T.
- *Plus tard :* autres modèles, réglage des hyperparamètres, courbe « performance en fonction de T », évaluation finale sur le test.

**Idées de prompts**

1. *Pipeline et baseline*
   > Build a scikit-learn Pipeline with a ColumnTransformer: numerical features → SimpleImputer(strategy="median") then StandardScaler; binary multi-hot features → passthrough. The pipeline must be fitted on TRAIN only. Models: DummyClassifier(strategy="most_frequent") and LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED). Evaluate on VALIDATION with average_precision_score (PR-AUC), roc_auc_score, F1, precision, recall and the confusion matrix. Also print the PR-AUC of a random classifier (= positive rate). Do not use the test set.

2. *Incertitude et seuil*
   > Compute a 95% bootstrap confidence interval (1000 resamples, seed fixed) for the validation PR-AUC and ROC-AUC of the logistic regression. Then explain why F1 depends on the 0.5 threshold, choose the threshold that maximizes F1 on validation, and report the metrics at this threshold. What assumptions does logistic regression make, and how can I check them here (linearity in log-odds, multicollinearity)?

**Erreurs de l'IA à surveiller**
- `StandardScaler` ou `SimpleImputer` ajustés **avant** le découpage, ou sur toutes les données : **note plafonnée à 10/20**.
- **L'accuracy** comme métrique principale avec des classes déséquilibrées.
- Un `roc_auc_score` calculé sur les classes prédites (0/1) au lieu des probabilités (`predict_proba`).
- Le test utilisé pour choisir le modèle ou le seuil.
- `random_state` oublié : résultats différents à chaque exécution.

**Questions possibles à l'oral :** pourquoi la PR-AUC plutôt que l'accuracy ? Que devient la PR-AUC si la proportion de succès change ? Que fait `class_weight="balanced"` ? Que se passe-t-il si on normalise avant le découpage ?

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
