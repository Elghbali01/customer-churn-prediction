# Étape 10 — Model Improvement

Cette étape améliore et sélectionne un candidat exclusivement à partir des 5 634 observations du train. Le test de 1 409 observations reste réservé. Toutes les transformations sont contenues dans le pipeline et réajustées dans chaque fold de `StratifiedKFold(5, shuffle=True, random_state=42)`.

## Déséquilibre et pondération

La cible (73,46 % No Churn, 26,54 % Churn) est déséquilibrée, mais pas extrême. Les pondérations testées sur Logistic Regression montrent qu'une pondération positive augmente fortement le Recall et le F1, au prix de la Precision et de l'Accuracy, sans améliorer le classement global.

| class_weight | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| `None` | 0.8064 | 0.6694 | 0.5344 | 0.5937 | 0.8466 | 0.6650 |
| `balanced` | 0.7536 | 0.5235 | 0.7980 | 0.6321 | 0.8466 | 0.6641 |
| `{0: 1, 1: 1.5}` | 0.7927 | 0.6017 | 0.6475 | 0.6235 | 0.8467 | 0.6648 |
| `{0: 1, 1: 2}` | 0.7797 | 0.5664 | 0.7251 | 0.6358 | 0.8467 | 0.6645 |

SMOTE n'est pas retenu : les pondérations et le seuil offrent déjà des leviers simples, le déséquilibre est modéré, et la majorité des variables sont catégorielles encodées. Ajouter un rééchantillonnage augmenterait la complexité et le risque d'une mauvaise génération synthétique sans bénéfice démontré. Aucune dépendance n'est ajoutée.

## Hyperparamètres et scoring

`average_precision` est le scoring principal : il évalue le classement de la classe positive et complète ROC-AUC lorsque cette classe est minoritaire. Accuracy, Precision, Recall, F1 et ROC-AUC sont conservés comme métriques secondaires.

Logistic Regression utilise une grille exhaustive de 27 configurations compatibles : `lbfgs` avec pénalité L2, cinq valeurs de C et trois pondérations ; `liblinear` avec L1/L2, trois valeurs de C et deux pondérations. Le meilleur réglage est `C=2`, `solver=lbfgs`, `penalty=l2`, `class_weight=None`.

Gradient Boosting utilise une recherche aléatoire bornée de 16 essais dans l'espace suivant : `n_estimators=[75,100,150]`, `learning_rate=[0.03,0.05,0.1]`, `max_depth=[1,2,3]`, `min_samples_split=[2,10]`, `min_samples_leaf=[1,10]`, `subsample=[0.8,1.0]`. Le meilleur réglage est `75`, `0.05`, `3`, `10`, `10`, `0.8`, respectivement.

| Candidat | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic baseline | 0.8064 ± 0.0090 | 0.6694 ± 0.0205 | 0.5344 ± 0.0385 | 0.5937 ± 0.0262 | 0.8466 ± 0.0118 | 0.6650 ± 0.0144 |
| Logistic tuned | 0.8064 ± 0.0099 | 0.6688 ± 0.0227 | 0.5358 ± 0.0397 | 0.5942 ± 0.0280 | 0.8468 ± 0.0119 | 0.6652 ± 0.0144 |
| Gradient baseline | 0.8025 ± 0.0104 | 0.6567 ± 0.0236 | 0.5365 ± 0.0250 | 0.5903 ± 0.0216 | 0.8479 ± 0.0118 | 0.6690 ± 0.0187 |
| Gradient tuned | 0.8060 ± 0.0104 | 0.6722 ± 0.0277 | 0.5258 ± 0.0215 | 0.5898 ± 0.0216 | 0.8513 ± 0.0114 | 0.6741 ± 0.0184 |

Le tuning de Logistic Regression est neutre. Gradient Boosting améliore modestement ses métriques de classement, mais réduit légèrement le Recall.

## Seuil sur probabilités out-of-fold

Les probabilités sont produites hors fold d'entraînement pour chaque observation du train. Le seuil candidat maximise le F1 sur la grille fixe 0,20–0,70 par pas de 0,02 ; les ex æquo privilégient Recall, puis Precision, puis le seuil le plus élevé.

| Seuil | Precision | Recall | F1 | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.4717 | 0.8642 | 0.6103 | 2692 | 1447 | 203 | 1292 |
| 0.24 | 0.5058 | 0.8201 | 0.6257 | 2941 | 1198 | 269 | 1226 |
| 0.28 | 0.5319 | 0.7813 | 0.6329 | 3111 | 1028 | 327 | 1168 |
| **0.30** | **0.5482** | **0.7572** | **0.6360** | **3206** | **933** | **363** | **1132** |
| 0.32 | 0.5592 | 0.7358 | 0.6355 | 3272 | 867 | 395 | 1100 |
| 0.40 | 0.5999 | 0.6468 | 0.6225 | 3494 | 645 | 528 | 967 |
| 0.50 | 0.6686 | 0.5358 | 0.5949 | 3742 | 397 | 694 | 801 |
| 0.60 | 0.7128 | 0.4000 | 0.5124 | 3898 | 241 | 897 | 598 |
| 0.70 | 0.7895 | 0.2508 | 0.3807 | 4039 | 100 | 1120 | 375 |

Le passage de 0,50 à 0,30 détecte 331 churners supplémentaires et évite 331 faux négatifs, mais génère 536 faux positifs supplémentaires. Ce seuil est un candidat opérationnel transparent, pas un optimum économique : les coûts métier ne sont pas disponibles.

## Feature Engineering et sélection

Avec le réglage Logistic retenu, les quatre features engineered apportent +0,0038 d'Average Precision, +0,0006 de ROC-AUC et +0,0165 de Precision, mais −0,0100 de Recall. Le gain est faible et mixte ; elles sont conservées car elles restent déterministes, interprétables et apportent un léger gain de classement. Aucune recherche combinatoire n'est effectuée.

Gradient Boosting dépasse Logistic Regression de 0,0089 en Average Precision moyenne. Conformément à la règle fixée avant la sélection (préférer Logistic si l'écart est inférieur ou égal à 0,01), le candidat retenu est le pipeline avec Feature Engineering et Logistic Regression (`C=2`, `lbfgs`, L2, sans class weights), avec un seuil opérationnel candidat de 0,30. Cette sélection train-only privilégie la simplicité et l'interprétabilité ; elle ne constitue pas une performance finale.

**Test set not consumed.**
