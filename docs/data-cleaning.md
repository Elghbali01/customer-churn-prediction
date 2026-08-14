# Étape 3 — Data Cleaning

Le nettoyage est exécuté par `python -m customer_churn_prediction.clean_data`. Il lit exclusivement le CSV brut, valide son empreinte, applique les règles documentées et écrit `data/processed/telco_customer_churn_clean.csv`. Le fichier raw n'est jamais réécrit.

## 1. `TotalCharges`

### Observation

`TotalCharges` est lu comme `object`. Onze valeurs sont des chaînes vides ou composées d'espaces, sans être reconnues comme manquantes par la lecture Pandas initiale.

### Investigation

Les 11 observations sont :

| customerID | tenure | MonthlyCharges | Contract | Churn |
| --- | ---: | ---: | --- | --- |
| 4472-LVYGI | 0 | 52.55 | Two year | No |
| 3115-CZMZD | 0 | 20.25 | Two year | No |
| 5709-LVOEQ | 0 | 80.85 | Two year | No |
| 4367-NUYAO | 0 | 25.75 | Two year | No |
| 1371-DWPAZ | 0 | 56.05 | Two year | No |
| 7644-OMVMY | 0 | 19.85 | Two year | No |
| 3213-VVOLG | 0 | 25.35 | Two year | No |
| 2520-SGTTA | 0 | 20.00 | Two year | No |
| 2923-ARZLG | 0 | 19.70 | One year | No |
| 4075-WKNIU | 0 | 73.35 | Two year | No |
| 2775-SEFEE | 0 | 61.90 | Two year | No |

Elles ont toutes `tenure == 0`. Cette caractéristique commune indique des clients sans période d'ancienneté révolue, donc sans montant cumulé antérieur, même si un tarif mensuel est déjà associé à leur compte.

### Décision

Attribuer `0.0` à `TotalCharges` pour ces 11 lignes. Cette valeur représente un cumul encore nul et conserve les nouveaux clients. Une moyenne ou une médiane inventerait un historique de facturation ; supprimer les lignes ferait perdre des clients valides.

### Transformation

Les chaînes vides sont d'abord converties en valeurs manquantes, puis toute la colonne est convertie numériquement. Seules les 11 valeurs associées à `tenure == 0` sont remplacées par `0.0`.

### Validation

`TotalCharges` est finalement de type `float64`, sans valeur manquante ni valeur négative.

## 2. `SeniorCitizen`

### Observation et investigation

La colonne est stockée en `int64` et contient uniquement `0` et `1`. Son rôle sémantique est celui d'un indicateur catégoriel binaire.

### Décision et transformation

Conserver `0/1` dans le dataset nettoyé. Transformer maintenant cette colonne en libellés n'améliorerait pas la qualité des données et anticiperait le preprocessing.

### Validation

Les deux valeurs d'origine sont préservées. La colonne devra être déclarée catégorielle lors du preprocessing ML.

## 3. Catégories liées à l'absence de service

### Observation et investigation

`No phone service` apparaît dans `MultipleLines` lorsque `PhoneService == No`. `No internet service` apparaît dans les six variables de services Internet lorsque `InternetService == No`. Ces valeurs décrivent l'inapplicabilité du service et non un simple refus d'option.

### Décision et transformation

Conserver ces catégories séparément de `No`. Aucune transformation n'est appliquée.

### Validation

Les cohérences entre service principal et catégories associées sont contrôlées par le pipeline.

## 4. `customerID`

### Observation et investigation

Les 7 043 identifiants sont présents, uniques et conformes au format observé `NNNN-AAAAA`.

### Décision et transformation

Conserver `customerID` sans modification pour la traçabilité. Il devra probablement être exclu des features au preprocessing ou à la sélection des variables.

### Validation

Aucun identifiant vide, mal formé ou dupliqué après nettoyage.

## 5. Target `Churn`

### Observation et décision

La target possède les modalités `Yes` et `No`. Elles sont conservées sans encodage, qui relève du preprocessing ML.

### Validation

La distribution reste strictement identique : 5 174 `No` et 1 869 `Yes`.

## 6. Autres contrôles de qualité

- Aucun doublon complet ou identifiant dupliqué.
- Aucun espace de début ou de fin hors des 11 valeurs de `TotalCharges`.
- Aucune variante de catégorie causée par la casse ou des espaces n'a été détectée.
- `tenure` est compris entre 0 et 72.
- `MonthlyCharges` est compris entre 18.25 et 118.75.
- Aucune valeur négative dans `tenure`, `MonthlyCharges` ou `TotalCharges` après nettoyage.
- Aucune valeur manquante après nettoyage.

Ces contrôles ne constituent pas une EDA et aucune valeur n'est modifiée en dehors de la règle validée pour `TotalCharges`.
