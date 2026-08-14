# Étape 5 — Preprocessing

## Objectif et périmètre

Cette documentation décrit l'infrastructure scikit-learn reproductible, désormais précédée par le Feature Engineering déterministe de l'Étape 6. Aucun modèle n'est entraîné, aucune performance n'est calculée et aucune matrice transformée n'est sauvegardée.

Le module de référence est `src/customer_churn_prediction/preprocessing.py`.

## Séparation des features et de la target

La target est `Churn`. Ses seules modalités autorisées sont validées avant encodage :

| Modalité | Valeur encodée |
| --- | ---: |
| `No` | 0 |
| `Yes` | 1 |

Avant Feature Engineering, `X` contient 7 043 lignes et 19 features. Le transformer déterministe ajoute ensuite 4 features, soit 23 avant le `ColumnTransformer`. `y` contient 7 043 valeurs binaires.

`customerID` est exclu uniquement lors de la construction de `X`. Il reste intact dans le dataset nettoyé pour assurer la traçabilité, mais sa valeur unique par client ne constitue pas une information prédictive généralisable.

## Typage sémantique explicite

### Features numériques

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `total_services` après Feature Engineering

### Features catégorielles

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `tenure_group` après Feature Engineering
- `contract_tenure` après Feature Engineering
- `internet_contract` après Feature Engineering

`SeniorCitizen` est explicitement catégorielle malgré son stockage entier `0/1`. Les listes ne dépendent donc pas d'une inférence fragile à partir des dtypes Pandas.

Les validations garantissent que les deux groupes sont disjoints, couvrent exactement les colonnes explicatives, ne contiennent aucun doublon et excluent `customerID` et `Churn`.

## Protocole train/test initial

Le split utilise :

- 80 % train et 20 % test ;
- `random_state=42` ;
- `stratify=y`.

| Ensemble | Lignes | No | Yes | Proportion Yes |
| --- | ---: | ---: | ---: | ---: |
| Global | 7 043 | 5 174 | 1 869 | 26,5370 % |
| Train | 5 634 | 4 139 | 1 495 | 26,5353 % |
| Test | 1 409 | 1 035 | 374 | 26,5436 % |

Le test set est réservé à l'évaluation finale. Une future cross-validation devra être effectuée uniquement sur le train set.

## Pipeline numérique

Le pipeline numérique contient uniquement :

```text
StandardScaler()
```

Le scaling rend le preprocessing compatible avec des modèles sensibles à l'échelle, notamment une future régression logistique. Il apprend moyenne et écart-type uniquement sur le train set.

Le dataset nettoyé ne contient aucune valeur manquante ; aucune imputation n'est donc ajoutée sans besoin réel. Le pipeline échouera explicitement si le contrat de données n'est plus respecté. Une stratégie d'imputation pourra être introduite plus tard si les données de production le justifient, après définition métier.

## Pipeline catégoriel

Le pipeline catégoriel contient :

```text
OneHotEncoder(handle_unknown="ignore", sparse_output=True)
```

Les catégories sont apprises uniquement sur le train set. Une catégorie inconnue au test ou lors d'une future prédiction produit des zéros pour les indicateurs connus de cette variable au lieu de provoquer une erreur.

## `ColumnTransformer`

Après le Feature Engineering, le transformer combine :

- `numeric` : pipeline numérique appliqué aux 4 colonnes explicites ;
- `categorical` : pipeline catégoriel appliqué aux 19 colonnes explicites ;
- `remainder="drop"` : toute colonne non déclarée est rejetée.

La représentation ajustée sur le split actuel possède 72 colonnes : 4 numériques standardisées et 68 indicateurs One-Hot. Les dimensions sont :

- train transformé : `(5634, 72)` ;
- test transformé : `(1409, 72)`.

Ces dimensions dépendent des catégories rencontrées sur le train et ne sont pas codées en dur dans le module.

## Prévention du data leakage

L'ordre des opérations est obligatoire :

1. charger le dataset nettoyé ;
2. construire `X` et `y` sans transformation apprenante ;
3. réaliser le split stratifié ;
4. construire le pipeline Feature Engineering + preprocessor non ajusté ;
5. appeler `fit` uniquement avec `X_train` et `y_train` ; le Feature Engineering n'apprend aucun paramètre ;
6. utiliser ce même objet pour transformer train et test.

Le test vérifie notamment que les moyennes apprises par `StandardScaler` correspondent exactement à `X_train`, et non au dataset complet.

## Intégration future

Le `ColumnTransformer` pourra être intégré sans modification dans :

```python
Pipeline([
    ("feature_engineering", ChurnFeatureEngineer()),
    ("preprocessor", build_preprocessor()),
    ("model", future_model),
])
```

Le modèle n'est volontairement pas défini à cette étape. Le pipeline complet sera ajusté dans les futures étapes Baseline et Machine Learning, ce qui préservera la frontière contre le leakage pendant la cross-validation.

## Artefacts

Aucune matrice transformée, aucun preprocessor ajusté et aucun objet sérialisé n'est sauvegardé. Le dataset nettoyé reste indépendant de la représentation propre à un modèle.
