# Étape 2 — Rapport de Data Understanding

> Ce rapport détecte les caractéristiques et problèmes du fichier brut. Il n'applique aucun nettoyage ni preprocessing.

## Périmètre et source inspectée

- Fichier : `data/raw/Telco-Customer-Churn.csv`
- Dimensions : **7043 lignes × 21 colonnes**
- Identifiant candidat : `customerID`
- Target : `Churn`

## Colonnes

`customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`

## Premières observations

| index | customerID | gender | SeniorCitizen | Partner | Dependents | tenure | PhoneService | MultipleLines | InternetService | OnlineSecurity | OnlineBackup | DeviceProtection | TechSupport | StreamingTV | StreamingMovies | Contract | PaperlessBilling | PaymentMethod | MonthlyCharges | TotalCharges | Churn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 7590-VHVEG | Female | 0 | Yes | No | 1 | No | No phone service | DSL | No | Yes | No | No | No | No | Month-to-month | Yes | Electronic check | 29.85 | 29.85 | No |
| 1 | 5575-GNVDE | Male | 0 | No | No | 34 | Yes | No | DSL | Yes | No | Yes | No | No | No | One year | No | Mailed check | 56.95 | 1889.5 | No |
| 2 | 3668-QPYBK | Male | 0 | No | No | 2 | Yes | No | DSL | Yes | Yes | No | No | No | No | Month-to-month | Yes | Mailed check | 53.85 | 108.15 | Yes |
| 3 | 7795-CFOCW | Male | 0 | No | No | 45 | No | No phone service | DSL | Yes | No | Yes | Yes | No | No | One year | No | Bank transfer (automatic) | 42.3 | 1840.75 | No |
| 4 | 9237-HQITU | Female | 0 | No | No | 2 | Yes | No | Fiber optic | No | No | No | No | No | No | Month-to-month | Yes | Electronic check | 70.7 | 151.65 | Yes |

## Types Pandas observés

| index | dtype |
| --- | --- |
| customerID | object |
| gender | object |
| SeniorCitizen | int64 |
| Partner | object |
| Dependents | object |
| tenure | int64 |
| PhoneService | object |
| MultipleLines | object |
| InternetService | object |
| OnlineSecurity | object |
| OnlineBackup | object |
| DeviceProtection | object |
| TechSupport | object |
| StreamingTV | object |
| StreamingMovies | object |
| Contract | object |
| PaperlessBilling | object |
| PaymentMethod | object |
| MonthlyCharges | float64 |
| TotalCharges | object |
| Churn | object |

## Informations générales (`DataFrame.info`)

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   object 
 20  Churn             7043 non-null   object 
dtypes: float64(1), int64(2), object(18)
memory usage: 1.1+ MB
```

## Statistiques descriptives

### Variables numériques

| index | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SeniorCitizen | 7043.0 | 0.1621468124378816 | 0.3686116056100131 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| tenure | 7043.0 | 32.37114865824223 | 24.55948102309446 | 0.0 | 9.0 | 29.0 | 55.0 | 72.0 |
| MonthlyCharges | 7043.0 | 64.76169246059918 | 30.090047097678493 | 18.25 | 35.5 | 70.35 | 89.85 | 118.75 |

### Variables non numériques

| index | count | unique | top | freq |
| --- | --- | --- | --- | --- |
| customerID | 7043 | 7043 | 7590-VHVEG | 1 |
| gender | 7043 | 2 | Male | 3555 |
| Partner | 7043 | 2 | No | 3641 |
| Dependents | 7043 | 2 | No | 4933 |
| PhoneService | 7043 | 2 | Yes | 6361 |
| MultipleLines | 7043 | 3 | No | 3390 |
| InternetService | 7043 | 3 | Fiber optic | 3096 |
| OnlineSecurity | 7043 | 3 | No | 3498 |
| OnlineBackup | 7043 | 3 | No | 3088 |
| DeviceProtection | 7043 | 3 | No | 3095 |
| TechSupport | 7043 | 3 | No | 3473 |
| StreamingTV | 7043 | 3 | No | 2810 |
| StreamingMovies | 7043 | 3 | No | 2785 |
| Contract | 7043 | 3 | Month-to-month | 3875 |
| PaperlessBilling | 7043 | 2 | Yes | 4171 |
| PaymentMethod | 7043 | 4 | Electronic check | 2365 |
| TotalCharges | 7043 | 6531 | 20.2 | 11 |
| Churn | 7043 | 2 | No | 5174 |

## Nombre de valeurs uniques

| index | n_unique_including_na |
| --- | --- |
| customerID | 7043 |
| gender | 2 |
| SeniorCitizen | 2 |
| Partner | 2 |
| Dependents | 2 |
| tenure | 73 |
| PhoneService | 2 |
| MultipleLines | 3 |
| InternetService | 3 |
| OnlineSecurity | 3 |
| OnlineBackup | 3 |
| DeviceProtection | 3 |
| TechSupport | 3 |
| StreamingTV | 3 |
| StreamingMovies | 3 |
| Contract | 3 |
| PaperlessBilling | 2 |
| PaymentMethod | 4 |
| MonthlyCharges | 1585 |
| TotalCharges | 6531 |
| Churn | 2 |

## Valeurs manquantes et chaînes vides

| index | missing_detected_by_pandas | blank_or_whitespace_strings |
| --- | --- | --- |
| customerID | 0 | 0 |
| gender | 0 | 0 |
| SeniorCitizen | 0 | 0 |
| Partner | 0 | 0 |
| Dependents | 0 | 0 |
| tenure | 0 | 0 |
| PhoneService | 0 | 0 |
| MultipleLines | 0 | 0 |
| InternetService | 0 | 0 |
| OnlineSecurity | 0 | 0 |
| OnlineBackup | 0 | 0 |
| DeviceProtection | 0 | 0 |
| TechSupport | 0 | 0 |
| StreamingTV | 0 | 0 |
| StreamingMovies | 0 | 0 |
| Contract | 0 | 0 |
| PaperlessBilling | 0 | 0 |
| PaymentMethod | 0 | 0 |
| MonthlyCharges | 0 | 0 |
| TotalCharges | 0 | 11 |
| Churn | 0 | 0 |

## Distribution brute de la target

| Churn | count | proportion |
| --- | --- | --- |
| No | 5174 | 0.7346301292063041 |
| Yes | 1869 | 0.2653698707936959 |

## Doublons potentiels

- Lignes entièrement dupliquées : **0**
- Valeurs dupliquées de `customerID` : **0**

## Observations

- `TotalCharges` est lu comme `object`, bien qu'il représente des montants.
- `TotalCharges` contient **11** chaînes vides ou composées d'espaces ; Pandas ne les compte pas comme valeurs manquantes avec la lecture utilisée.
- `SeniorCitizen` est lu comme entier avec deux valeurs distinctes ; son rôle semble être celui d'un indicateur catégoriel.
- Certaines colonnes de services possèdent des catégories telles que `No internet service` ou `No phone service`, distinctes de `No`.
- La target contient deux modalités brutes, `Yes` et `No`.
- Le fichier sélectionné ne contient pas les champs enrichis connus `Churn Score`, `Churn Reason`, `Churn Category`, `Customer Status` ou `Churn Value`.

## Décisions futures — non appliquées à cette étape

- Définir le traitement de `TotalCharges` et de ses chaînes vides lors du Data Cleaning.
- Confirmer le typage métier de `SeniorCitizen`.
- Décider du traitement de l'identifiant avant la modélisation.
- Définir et documenter l'encodage de la target et des catégories lors du preprocessing.
- Réévaluer toute colonne candidate au regard de sa disponibilité au moment de la prédiction et du target leakage.
