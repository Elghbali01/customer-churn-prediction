# Décisions du projet

## Objectif général

Construire de manière incrémentale et reproductible un système complet de prédiction du churn client, sans anticiper les choix qui dépendent des données ou des résultats expérimentaux.

## Architecture initiale

- `data/` sépare les données sources (`raw`) des données transformées (`processed`).
- `notebooks/` accueillera les analyses exploratoires et expérimentations.
- `src/customer_churn_prediction/` contient le code Python réutilisable.
- `models/` est réservé aux artefacts de modèles générés.
- `api/` et `app/` sont réservés aux futures interfaces FastAPI et Streamlit.
- `tests/` accueillera les tests automatisés.
- `reports/figures/` contient les visualisations générées pour les rapports.

## Conventions et décisions techniques

- Python 3.11 constitue la version minimale prise en charge.
- La disposition `src` évite les imports accidentels depuis la racine du dépôt.
- Les dépendances sont centralisées dans `pyproject.toml` et seront ajoutées au moment où elles deviennent nécessaires.
- Les données, modèles et figures générés ne sont pas versionnés par défaut ; seuls leurs dossiers sont conservés via `.gitkeep`.
- Aucun choix de dataset, de modèle ou de métrique n'est arrêté pendant l'initialisation.

## Étape 1 — Business Understanding

- Le cas d'usage est la priorisation des actions de rétention pour des clients télécom à partir d'un risque estimé de churn.
- Le problème Data Science est formulé comme un apprentissage supervisé de classification binaire : `Churn` est la classe positive et `No Churn` la classe négative.
- La sortie attendue associe une probabilité de churn à une décision de classification fondée sur un seuil qui reste à déterminer.
- L'accuracy ne sera pas utilisée seule. Le Recall, la Precision, le F1-score, le ROC-AUC et la matrice de confusion seront étudiés ultérieurement.
- Le choix de la métrique principale et du seuil dépendra du compromis métier entre churners détectés et fausses alertes ; aucun objectif chiffré n'est fixé.
- Les variables, le dataset, les coûts économiques et la définition opérationnelle précise du churn restent à valider.

## Étape 2 — Dataset Acquisition & Data Understanding

- La variante IBM classique `Telco-Customer-Churn.csv` à 21 colonnes est retenue pour sa table client directement adaptée à la classification binaire et sa target `Churn`.
- La copie publiée dans le dépôt officiel IBM `telco-customer-churn-on-icp4d` est la source d'acquisition reproductible ; son empreinte SHA-256 est contrôlée.
- `customerID` est identifié comme identifiant client et `Churn` comme target (`Yes` positif, `No` négatif).
- `data/raw/` reste immuable : les problèmes détectés sont documentés mais ne sont pas corrigés à cette étape.
- Les champs enrichis `Customer Status`, `Churn Value`, `Churn Score`, `Churn Category` et `Churn Reason` sont considérés comme interdits par défaut en raison du risque de target leakage. Ils sont absents du fichier retenu.
- Le traitement de `TotalCharges`, des catégories, de l'identifiant et des types est reporté au Data Cleaning ou au preprocessing selon le cas.

## Étape 3 — Data Cleaning

- Les 11 `TotalCharges` vides correspondent tous à des clients avec `tenure == 0`. Ils sont remplacés par `0.0`, interprété comme un cumul de facturation encore nul, puis la colonne est stockée en `float64`.
- `SeniorCitizen` conserve son stockage `0/1` ; son rôle catégoriel binaire devra être explicité lors du preprocessing.
- `No internet service` et `No phone service` sont conservés comme catégories métier distinctes de `No`.
- `customerID` est conservé pour la traçabilité, mais devra probablement être exclu des features du modèle.
- `Churn` conserve ses modalités `Yes` et `No` ; son encodage est reporté au preprocessing.
- Aucune ligne n'est supprimée : aucun doublon complet ni identifiant dupliqué n'a été détecté.
- Le dataset nettoyé est reconstruit de manière déterministe sous `data/processed/` à partir du raw dont l'empreinte est validée.

## Étape 4 — Exploratory Data Analysis

- L'EDA utilise exclusivement `data/processed/telco_customer_churn_clean.csv` et ne modifie aucun dataset.
- Les analyses catégorielles rapportent systématiquement effectifs, churners et churn rates ; `SeniorCitizen` est traité sémantiquement comme catégoriel.
- Les croisements multivariés retenus sont contrat × ancienneté, service Internet × contrat, et charges mensuelles × contrat × churn.
- La matrice de corrélation est limitée aux variables quantitatives `tenure`, `MonthlyCharges` et `TotalCharges`.
- Huit figures ciblées sont générées sous `reports/figures/` et restent ignorées par Git comme artefacts reproductibles.
- Les écarts observés sont documentés comme associations propres au dataset, sans affirmation causale.
- Les idées issues de l'EDA restent des hypothèses : aucune feature, aucun encodage et aucune stratégie de rééquilibrage ne sont implémentés.

## Étape 5 — Preprocessing

- `Churn` est encodé explicitement avec `No -> 0` et `Yes -> 1` après validation des modalités.
- `customerID` est conservé dans le dataset nettoyé mais exclu de `X`.
- Les features numériques explicites sont `tenure`, `MonthlyCharges` et `TotalCharges`; les 16 autres variables explicatives, dont `SeniorCitizen`, sont catégorielles.
- Le protocole initial utilise un split 80/20 stratifié avec `random_state=42`; le test set est réservé à l'évaluation finale.
- Les variables numériques sont standardisées avec `StandardScaler` sans imputation, puisque le dataset nettoyé ne contient aucune valeur manquante.
- Les variables catégorielles utilisent `OneHotEncoder(handle_unknown="ignore")`.
- Le `ColumnTransformer` est ajusté uniquement sur le train set et pourra être inséré dans un futur pipeline `preprocessor -> model`.
- Aucune matrice transformée ni aucun preprocessor ajusté n'est sérialisé à cette étape.

## Étape 6 — Feature Engineering

- Quatre features déterministes sont ajoutées à la volée : `tenure_group`, `contract_tenure`, `internet_contract` et `total_services`.
- Les bandes d'ancienneté sont `0-12`, `13-24`, `25-48` et `49+`; elles reposent sur des périodes calendaires interprétables, non sur une optimisation de la target.
- `total_services` compte uniquement les modalités `Yes` parmi six services Internet; `No` et `No internet service` valent zéro sans être fusionnés dans les variables sources.
- Les 19 features originales restent conservées et `customerID` demeure exclu de `X`.
- Aucune feature dérivée des charges n'est créée : le ratio étudié est fortement redondant, ambigu historiquement et indéfini lorsque `tenure == 0`.
- Le pipeline devient `feature_engineering -> preprocessor`, puis pourra recevoir un modèle lors d'une étape ultérieure.
- Le Feature Engineering n'apprend aucun paramètre et n'utilise pas `Churn`; le scaler et l'encodeur restent ajustés uniquement sur le train.
- Aucune table enrichie ni aucun transformer ajusté n'est sérialisé.

## Étapes 7–9 — Baseline, Machine Learning et évaluation

- La baseline est `DummyClassifier(strategy="most_frequent")`.
- Les candidats comparés sont Logistic Regression, Decision Tree, Random Forest et Gradient Boosting, tous dans le même pipeline complet.
- La comparaison utilise exclusivement le train avec `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Accuracy, Precision, Recall, F1 et ROC-AUC sont rapportés en moyenne et écart-type ; `Churn = 1` reste la classe positive.
- La ROC-AUC moyenne CV sert uniquement à nommer un candidat provisoire ; Gradient Boosting arrive premier de très peu devant Logistic Regression.
- La comparaison du Feature Engineering utilise Logistic Regression, les mêmes folds et les mêmes paramètres ; les gains observés sont faibles et mixtes.
- Les matrices de confusion et courbes ROC sont calculées à partir de prédictions out-of-fold sur le train.
- Le test set n'est ni prédit ni évalué : `Test set not consumed.`
- XGBoost est reporté car les modèles scikit-learn suffisent pour ce premier benchmark.
- Aucun tuning, rééquilibrage, class weight, changement de seuil ou modèle final n'est décidé.
- Aucun modèle ou pipeline ajusté n'est sérialisé.

## Étape 10 — Model Improvement

- Toutes les décisions d'amélioration reposent uniquement sur le train, avec les mêmes cinq folds stratifiés et des probabilités out-of-fold pour le seuil ; le test reste réservé.
- `average_precision` est le scoring principal du tuning, tandis que les autres métriques restent rapportées.
- Les class weights améliorent Recall/F1 mais dégradent Precision/Accuracy sans gain de classement ; le candidat retenu reste sans pondération.
- SMOTE n'est pas ajouté : déséquilibre modéré, données largement catégorielles encodées, leviers intégrés suffisants et complexité non justifiée.
- Logistic Regression est explorée par une grille compatible et Gradient Boosting par 16 tirages bornés.
- Les quatre features engineered sont conservées : leur gain est faible et mixte, mais elles sont déterministes, interprétables et améliorent légèrement l'Average Precision.
- Lorsque l'écart d'Average Precision est inférieur ou égal à 0,01, la règle de sélection privilégie Logistic Regression. L'écart observé de Gradient Boosting est 0,0089.
- Le candidat train-only est Logistic Regression (`C=2`, `solver=lbfgs`, pénalité L2 par défaut, sans class weights), avec Feature Engineering.
- Le seuil opérationnel candidat 0,30 maximise le F1 OOF sur la grille étudiée. Il n'est pas présenté comme économiquement optimal.
- Aucun modèle n'est sérialisé et aucune évaluation test ou explicabilité n'est lancée.

## Étapes 11–12 — Explainability et pipeline final

- Le candidat de l'Étape 10 a été figé avant consommation du test : Logistic Regression (`C=2`, `lbfgs`, L2, sans class weights), quatre features engineered et seuil `0,30`.
- Le pipeline a été ajusté une seule fois sur les 5 634 lignes du train. Le test de 1 409 lignes a été consommé uniquement pour l'évaluation finale ; aucune sélection post-test n'a été effectuée.
- Au seuil 0,30, les résultats test sont Accuracy `0,7608`, Precision `0,5347`, Recall `0,7620`, F1 `0,6284`, avec TN=787, FP=248, FN=89 et TP=285. ROC-AUC=`0,8429`, AP=`0,6379`.
- Les 72 coefficients transformés et leurs odds ratios sont exportés. Leur interprétation est conditionnelle au modèle et ne constitue pas une relation causale.
- SHAP utilise `LinearExplainer` sur le modèle final et le mapping exact des 72 features transformées. Les importances expliquent les prédictions, non les causes du churn.
- Le pipeline complet est sérialisé dans `models/churn_pipeline.joblib` et son seuil/métadonnées dans `models/model_metadata.json`.
- La validation round-trip produit des probabilités et classes identiques, avec une différence maximale de `0,0`.
- `predict_churn` constitue le contrat de prédiction réutilisable pour une future API, sans démarrer FastAPI.

## Étapes 13–15 — Architecture, FastAPI et tests

- FastAPI expose le pipeline sérialisé existant ; aucune logique de Feature Engineering ou de preprocessing n'est dupliquée dans les endpoints.
- Le pipeline et ses métadonnées sont chargés une seule fois pendant le lifespan de l'application, depuis des chemins ancrés sur la racine du projet.
- Pydantic valide exactement les 19 features brutes, refuse les champs supplémentaires et limite les catégories aux modalités historiques.
- Deux règles inter-variables sont retenues : cohérence téléphonie/`MultipleLines` et Internet/six add-ons. Elles reproduisent des contraintes explicites du dataset.
- Le seuil reste `0,30` et provient des métadonnées validées au démarrage.
- `/predict/batch` est retenu avec une limite de 100 clients, ordre préservé et absence de stockage.
- Aucun payload n'est stocké ou loggué par défaut ; aucune base de données, authentification ou collecte utilisateur n'est ajoutée.
- Docker, déploiement, CI/CD et monitoring restent reportés.

## Étapes 16–17 — Docker et préparation du déploiement

- L'image de base est `python:3.11-slim-bookworm`, compatible Python 3.11 et plus légère qu'une image Python complète.
- Scikit-learn reste verrouillé sur `>=1.8,<1.9` afin de garantir la compatibilité avec l'artefact sérialisé en 1.8.0.
- Le Dockerfile reste mono-stage : aucun artefact compilé séparé ne justifie un multi-stage pour cette petite application Python.
- L'application s'exécute avec un utilisateur système non-root et un healthcheck `/health` fondé sur la bibliothèque standard Python.
- Uvicorn écoute sur `0.0.0.0` et `${PORT:-8000}`, sans reload ni access log de payload.
- Docker Compose est rejeté : l'application est mono-service, sans base de données, Redis ou queue.
- `.dockerignore` exclut analyses, tests, données et caches, tout en conservant explicitement le pipeline et ses métadonnées.
- Les deux artefacts ML sont explicitement autorisés par `.gitignore` car ils sont indispensables au build depuis Git et aucun téléchargement dynamique n'est prévu.
- Render est retenu comme cible portfolio Docker/HTTPS à maintenance limitée ; `render.yaml` configure un Web Service gratuit et `/health`.
- Aucun compte Render n'étant disponible, le statut reste `DEPLOYMENT-READY — manual account action required` et aucune URL n'est inventée.
- CI/CD, monitoring, drift detection et retraining automatique restent reportés.

## Validation Docker et User Interface

- Docker Desktop 29.7.2 a validé le build, le run, le healthcheck, les endpoints, les 422, le hash et la parité locale/conteneur.
- L'interface utilise HTML/CSS/JavaScript natifs servis par FastAPI afin de conserver un seul service et aucun toolchain Node.
- `/` sert désormais l'interface ; `/docs` reste la documentation technique Swagger et les endpoints API sont inchangés.
- Les 19 features brutes sont organisées en cinq sections et les dépendances téléphone/Internet sont synchronisées côté interface, puis revalidées par Pydantic.
- Les niveaux Low (`<0,30`), Moderate (`0,30–0,60`) et High (`≥0,60`) sont explicitement des catégories UX ; le seuil ML reste `0,30`.
- Les fichiers statiques sont embarqués automatiquement par la copie du dossier `api/` dans l'image.
- La validation navigateur desktop et mobile n'ajoute aucune dépendance Playwright/Selenium au projet.
- Render reste correctement configuré ; le déploiement nécessite toujours une action de compte manuelle.
