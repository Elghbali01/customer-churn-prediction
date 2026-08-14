# Étape 4 — Rapport d'Exploratory Data Analysis

## Objectif et méthodologie

Cette EDA décrit la target, les distributions numériques, les churn rates catégoriels et trois croisements ciblés à partir de `data/processed/telco_customer_churn_clean.csv`. Elle ne modifie aucune donnée, n'encode aucune variable et n'entraîne aucun modèle.

Les résultats décrivent des **associations observées** dans le sample IBM fictif. Ils ne permettent pas de conclure à des relations causales.

## Distribution de la target

| Churn | Clients | Proportion |
| --- | ---: | ---: |
| No | 5 174 | 73,46 % |
| Yes | 1 869 | 26,54 % |

La classe Churn est minoritaire. L'accuracy seule pourrait donc être trompeuse, mais aucune stratégie de rééquilibrage n'est décidée pendant l'EDA.

![Distribution de Churn](figures/churn_target_distribution.png)

## Variables numériques

| Variable | Ensemble — moyenne | Churn Yes — moyenne / médiane | Churn No — moyenne / médiane |
| --- | ---: | ---: | ---: |
| `tenure` | 32,37 | 17,98 / 10,00 | 37,57 / 38,00 |
| `MonthlyCharges` | 64,76 | 74,44 / 79,65 | 61,27 / 64,43 |
| `TotalCharges` | 2 279,73 | 1 531,80 / 703,55 | 2 549,91 / 1 679,52 |

Les churners ont une ancienneté plus courte et des charges mensuelles plus élevées dans les comparaisons brutes. Leur total cumulé est plus faible, résultat cohérent avec une ancienneté plus courte. Les distributions étendues et asymétriques ne justifient pas à elles seules de supprimer les valeurs extrêmes.

![Distributions numériques](figures/numeric_distributions_by_churn.png)

![Boxplots numériques](figures/numeric_boxplots_by_churn.png)

## Variables catégorielles prioritaires

### Contrat

| Contract | Clients | Churners | Churn rate |
| --- | ---: | ---: | ---: |
| Month-to-month | 3 875 | 1 655 | 42,71 % |
| One year | 1 473 | 166 | 11,27 % |
| Two year | 1 695 | 48 | 2,83 % |

### Service Internet

| InternetService | Clients | Churners | Churn rate |
| --- | ---: | ---: | ---: |
| Fiber optic | 3 096 | 1 297 | 41,89 % |
| DSL | 2 421 | 459 | 18,96 % |
| No | 1 526 | 113 | 7,40 % |

### Support technique et sécurité en ligne

| Variable / catégorie | Clients | Churners | Churn rate |
| --- | ---: | ---: | ---: |
| TechSupport — No | 3 473 | 1 446 | 41,64 % |
| TechSupport — Yes | 2 044 | 310 | 15,17 % |
| OnlineSecurity — No | 3 498 | 1 461 | 41,77 % |
| OnlineSecurity — Yes | 2 019 | 295 | 14,61 % |
| No internet service | 1 526 | 113 | 7,40 % |

### Paiement et facturation

| Variable / catégorie | Clients | Churners | Churn rate |
| --- | ---: | ---: | ---: |
| Electronic check | 2 365 | 1 071 | 45,29 % |
| Mailed check | 1 612 | 308 | 19,11 % |
| Bank transfer (automatic) | 1 544 | 258 | 16,71 % |
| Credit card (automatic) | 1 522 | 232 | 15,24 % |
| PaperlessBilling — Yes | 4 171 | 1 400 | 33,57 % |
| PaperlessBilling — No | 2 872 | 469 | 16,33 % |

![Churn rates des catégories prioritaires](figures/key_categorical_churn_rates.png)

## Insights structurés

### Contrat et ancienneté

**Question →** Le churn associé au contrat varie-t-il avec l'ancienneté ?

**Analyse →** Croisement de quatre bandes d'ancienneté avec les trois types de contrat.

**Résultat quantitatif →** Les clients en contrat mensuel et avec 0–12 mois d'ancienneté présentent un churn rate de **51,35 %** (1 024 churners sur 1 994). Le taux du contrat mensuel diminue à **26,02 %** pour 49–72 mois. Les contrats de deux ans restent entre **0 % et 3,33 %** selon la bande.

**Interprétation métier →** Les nouveaux clients en contrat flexible constituent un segment pertinent à surveiller pour de futures actions de rétention.

**Prudence/limite →** Le contrat, l'ancienneté, les services et les charges sont liés entre eux ; ce croisement ne démontre pas qu'un changement de contrat réduirait causalement le churn.

![Contrat et ancienneté](figures/contract_tenure_churn_heatmap.png)

### Internet et contrat

**Question →** Certains couples service Internet / contrat sont-ils associés à un churn élevé ?

**Analyse →** Calcul des effectifs et churn rates pour les neuf combinaisons.

**Résultat quantitatif →** Fibre + contrat mensuel présente le taux le plus élevé : **54,61 %** (1 162 sur 2 128). DSL + contrat mensuel atteint **32,22 %**, contre **18,89 %** sans Internet + contrat mensuel. Fibre + contrat deux ans est à **7,23 %**.

**Interprétation métier →** Le segment fibre sous contrat mensuel mérite une analyse ultérieure des offres, de l'expérience et des tarifs.

**Prudence/limite →** Le dataset ne fournit pas de preuve sur le mécanisme à l'origine de cet écart et ne permet pas une conclusion causale.

![Internet et contrat](figures/internet_contract_churn_heatmap.png)

### Charges mensuelles, churn et contrat

**Question →** Les différences de charges mensuelles persistent-elles au sein des contrats ?

**Analyse →** Comparaison des moyennes et médianes par contrat et target.

**Résultat quantitatif →** En contrat mensuel, la moyenne est de **73,02** chez les churners contre **61,46** chez les non-churners. Pour un an : **85,05** contre **62,51**. Pour deux ans : **86,78** contre **60,01**, mais seulement 48 churners composent ce dernier groupe.

**Interprétation métier →** Les charges mensuelles restent associées au churn dans chaque type de contrat et pourront être informatives pour la modélisation.

**Prudence/limite →** Les offres et services souscrits influencent probablement les charges ; la comparaison brute n'isole pas un effet propre du prix.

![Charges, contrat et churn](figures/monthly_charges_contract_churn.png)

### Support et sécurité

**Question →** L'absence de support ou de sécurité est-elle associée au churn ?

**Analyse →** Comparaison des churn rates bruts par statut de service.

**Résultat quantitatif →** Sans support technique, le taux est **41,64 %**, contre **15,17 %** avec support. Sans sécurité en ligne, il est **41,77 %**, contre **14,61 %** avec sécurité.

**Interprétation métier →** La présence de services d'accompagnement peut signaler des profils de clients différents et mérite d'être représentée dans les futurs modèles.

**Prudence/limite →** La souscription à ces services dépend du type d'accès, du contrat et du profil client ; aucune causalité n'est établie.

### Paiement et facturation dématérialisée

**Question →** Les modalités de paiement et de facturation distinguent-elles des groupes de churn ?

**Analyse →** Churn rates par méthode de paiement et statut PaperlessBilling.

**Résultat quantitatif →** Le chèque électronique atteint **45,29 %**, contre **15,24 %** pour la carte automatique. La facturation dématérialisée atteint **33,57 %**, contre **16,33 %** sans celle-ci.

**Interprétation métier →** Ces modalités peuvent aider à identifier des profils à risque et des parcours de paiement à étudier.

**Prudence/limite →** Ces catégories peuvent surtout refléter d'autres différences de contrat, d'ancienneté ou de comportement.

## Autres variables catégorielles

- Senior : **41,68 %** contre **23,61 %** pour les non-seniors.
- Sans partenaire : **32,96 %** contre **19,66 %** avec partenaire.
- Sans personnes à charge : **31,28 %** contre **15,45 %** avec personnes à charge.
- Genre : **26,92 %** pour les femmes et **26,16 %** pour les hommes, soit un écart brut faible.
- PhoneService : **26,71 %** avec service et **24,93 %** sans service, également un écart faible.
- OnlineBackup absent : **39,93 %**, contre **21,53 %** avec le service.
- DeviceProtection absent : **39,13 %**, contre **22,50 %** avec le service.

## Corrélations numériques

| Variables | Corrélation de Pearson |
| --- | ---: |
| tenure / TotalCharges | 0,826 |
| MonthlyCharges / TotalCharges | 0,651 |
| tenure / MonthlyCharges | 0,248 |

`SeniorCitizen` est volontairement absent de cette matrice puisqu'il s'agit sémantiquement d'une catégorie binaire. Une corrélation ne démontre aucune causalité et aucune variable n'est supprimée à cette étape.

![Corrélations numériques](figures/numeric_correlations.png)

## Hypothèses de feature engineering — non implémentées

- bandes d'ancienneté ;
- interaction contrat × ancienneté ;
- interaction Internet × contrat ;
- indicateur de présence de support/sécurité ;
- nombre de services souscrits ;
- relation entre charges mensuelles et ancienneté ;
- regroupements de moyens de paiement automatiques et non automatiques, seulement si justifiés ultérieurement.

## Limites

- Entreprise et clients fictifs ; généralisation externe inconnue.
- Données observationnelles sans preuve de causalité.
- Une photographie client, sans historique événementiel détaillé ni horizon temporel explicite.
- Comparaisons brutes non ajustées pour les facteurs confondants.
- Certains segments croisés ont de petits effectifs.
- Variables de services potentiellement redondantes.
- Aucun coût métier disponible et aucune conclusion sur le seuil de décision.
- Aucune performance ML ne peut être déduite de l'EDA.

## Reproductibilité des figures

Les huit figures sont générées avec :

```powershell
$env:PYTHONPATH = (Resolve-Path src)
python -m customer_churn_prediction.eda
```

Les PNG restent ignorés par Git conformément à la stratégie initiale sur les artefacts générés. Le notebook et le présent rapport sont versionnables.
