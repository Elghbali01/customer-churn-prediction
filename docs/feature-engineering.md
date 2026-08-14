# Étape 6 — Feature Engineering

## Objectif et architecture

Le Feature Engineering ajoute quatre variables interprétables issues des observations de l'EDA. Il est implémenté par `ChurnFeatureEngineer`, un transformer scikit-learn déterministe qui ne consulte jamais `Churn` et n'apprend aucun paramètre statistique.

L'ordre prévu pour les futurs modèles est :

```text
ChurnFeatureEngineer → ColumnTransformer → futur modèle
```

Le split train/test est réalisé avant l'ajustement du pipeline. Le transformer produit les mêmes règles sur train, test et futures observations. Le `ColumnTransformer`, qui apprend les paramètres de scaling et les catégories, reste ajusté uniquement sur le train.

## Features créées

### 1. `tenure_group`

- **Sources :** `tenure`
- **Règle :** `0–12`, `13–24`, `25–48`, `49+` mois
- **Type final :** catégorielle
- **Justification métier :** distinguer première année, deuxième année, ancienneté intermédiaire de deux à quatre ans et clients établis au-delà de quatre ans.
- **Justification EDA :** le churn variait fortement selon l'ancienneté, particulièrement parmi les clients récents.
- **Limites :** la discrétisation perd de la précision ; `tenure` numérique est donc conservé. Les seuils sont des périodes calendaires simples, non optimisées sur `Churn`.

Les intervalles sont fermés à droite : 12 appartient à `0-12`, 24 à `13-24` et 48 à `25-48`. Les anciennetés supérieures sont acceptées dans `49+`; les valeurs négatives, manquantes ou non finies sont refusées.

### 2. `contract_tenure`

- **Sources :** `Contract`, `tenure_group`
- **Règle :** concaténation `Contract + " | " + tenure_group`
- **Exemple :** `Month-to-month | 0-12`
- **Type final :** catégorielle
- **Justification métier :** représenter conjointement l'engagement contractuel et le stade de la relation client.
- **Justification EDA :** le segment contrat mensuel et 0–12 mois présentait un churn rate brut de 51,35 %, contre des taux plus faibles pour les anciennetés supérieures et les contrats longs.
- **Limites :** l'interaction augmente le nombre de modalités et ne démontre aucune causalité. Les deux variables sources sont conservées.

### 3. `internet_contract`

- **Sources :** `InternetService`, `Contract`
- **Règle :** concaténation `InternetService + " | " + Contract`
- **Exemple :** `Fiber optic | Month-to-month`
- **Type final :** catégorielle
- **Justification métier :** associer l'offre d'accès à son niveau d'engagement.
- **Justification EDA :** fibre + contrat mensuel présentait le churn rate brut le plus élevé du croisement étudié, 54,61 %.
- **Limites :** la variable représente une association segmentaire et augmente la dimension One-Hot. Les sources restent disponibles séparément.

### 4. `total_services`

- **Sources :** `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Règle :** nombre de valeurs exactement égales à `Yes` parmi les six colonnes
- **Domaine :** entier de 0 à 6
- **Type final :** numérique (`int8` avant preprocessing)
- **Justification métier :** fournir une mesure simple de profondeur d'équipement en services Internet.
- **Justification EDA :** plusieurs services, notamment support, sécurité, backup et protection, présentaient des churn rates différents selon la souscription.
- **Limites :** tous les services reçoivent le même poids et le compteur ne capture pas leur nature. Les six colonnes originales sont conservées.

`No` et `No internet service` comptent tous deux pour zéro, mais restent distincts dans leurs variables sources. Toute autre modalité provoque une erreur explicite.

## Feature de charges étudiée mais rejetée

La candidate étudiée était un rapport du type :

```text
TotalCharges / (MonthlyCharges × tenure)
```

Sur les 7 032 clients avec `tenure > 0`, ce rapport possède une médiane de `1,0000`, une moyenne de `1,0003` et un écart-type de `0,0512`. `TotalCharges` est déjà corrélé à `tenure` à `0,8262` et à `MonthlyCharges` à `0,6512`.

La feature n'est pas créée car elle serait largement redondante, sensible aux variations tarifaires historiques difficiles à interpréter et indéfinie pour les 11 clients avec `tenure == 0`. Les trois variables de charges et d'ancienneté restent disponibles ; une éventuelle dérivée pourra être réévaluée expérimentalement plus tard.

## Sélection des variables

- `customerID` reste exclu de `X`, mais présent dans le dataset nettoyé pour la traçabilité.
- `Churn` reste exclusivement la target.
- Les 19 features originales sont conservées.
- Les quatre nouvelles features sont ajoutées.
- Aucune variable n'est supprimée sur la seule base d'une intuition de redondance.

La table explicative passe donc de 19 à 23 features avant preprocessing.

## Preprocessing adapté

### Numériques — 4

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `total_services`

### Catégorielles — 19

Les 16 variables catégorielles précédentes, auxquelles s'ajoutent :

- `tenure_group`
- `contract_tenure`
- `internet_contract`

Sur le split actuel, l'encodage produit 72 colonnes : 4 numériques standardisées et 68 indicateurs One-Hot. Les dimensions observées sont `(5634, 72)` pour le train et `(1409, 72)` pour le test.

## Data leakage et déterminisme

Les quatre règles utilisent uniquement les caractéristiques présentes sur une observation et des seuils métier fixes. `fit` valide le schéma mais n'apprend ni seuil, ni moyenne, ni catégorie, et ignore complètement `y`.

Les tests vérifient que :

- inverser artificiellement `y` ne change aucune feature ;
- deux transformations du même DataFrame sont identiques ;
- le DataFrame source n'est pas modifié ;
- le split intervient avant le `fit` du pipeline ;
- le scaler et le One-Hot Encoder restent ajustés uniquement sur le train.

## Artefacts

Aucune table enrichie n'est écrite dans `data/processed/`. Les features sont reconstruites à la volée dans le pipeline afin d'éviter plusieurs versions persistées du dataset et de garantir les mêmes règles en entraînement et en inférence.
