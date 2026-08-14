# Étapes 7–9 — Baseline, Machine Learning et évaluation

## Protocole expérimental

Le dataset nettoyé est séparé une seule fois selon le protocole validé : 5 634 observations train et 1 409 observations test. Toutes les comparaisons de ce bloc utilisent exclusivement le train.

```text
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Dans chaque fold, le pipeline complet est réajusté :

```text
ChurnFeatureEngineer → ColumnTransformer → estimateur
```

La classe positive est `Churn = 1`. Le meilleur candidat provisoire est identifié par la moyenne ROC-AUC CV parmi les vrais modèles, car cette métrique évalue le classement des risques indépendamment du seuil par défaut. Precision, Recall et F1 restent indispensables à l'interprétation métier.

**Test set not consumed.** Il n'a servi ni au choix du modèle, ni aux métriques, ni aux figures, ni au Feature Engineering, ni au choix d'un seuil.

## Modèles et paramètres

### Dummy baseline

```python
DummyClassifier(strategy="most_frequent")
```

Cette baseline prédit toujours la classe majoritaire `No`. Elle représente le plancher que les vrais modèles doivent dépasser pour détecter les churners.

### Logistic Regression

```python
LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    C=1.0,
    penalty="l2",
    class_weight=None,
    random_state=42,
)
```

Modèle linéaire interprétable de référence. `max_iter` est augmenté pour permettre la convergence ; aucun tuning n'est réalisé.

### Decision Tree

```python
DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight=None,
    random_state=42,
)
```

Arbre simple non réglé, inclus comme famille non linéaire. Ses paramètres par défaut peuvent conduire à du surapprentissage ; ce comportement n'est pas corrigé dans ce bloc.

### Random Forest

```python
RandomForestClassifier(
    n_estimators=300,
    criterion="gini",
    max_depth=None,
    class_weight=None,
    random_state=42,
    n_jobs=-1,
)
```

Ensemble d'arbres réduisant la variance d'un arbre unique. Le nombre d'arbres vise une estimation stable, sans recherche d'hyperparamètres.

### Gradient Boosting

```python
GradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    max_depth=3,
    random_state=42,
)
```

Ensemble séquentiel d'arbres peu profonds. Tous les paramètres d'apprentissage restent ceux par défaut de scikit-learn, sans tuning.

### XGBoost

XGBoost est reporté. Les quatre familles scikit-learn suffisent pour établir le premier benchmark ; ajouter XGBoost introduirait une dépendance et une surface de tuning supplémentaires sans nécessité à cette étape.

## Métriques

- **Accuracy :** proportion totale de prédictions correctes ; trompeuse seule lorsque la classe majoritaire domine.
- **Precision :** parmi les clients prédits churners, proportion réellement churner.
- **Recall :** parmi les churners réels, proportion détectée.
- **F1 :** moyenne harmonique de Precision et Recall.
- **ROC-AUC :** capacité à classer les churners au-dessus des non-churners sur l'ensemble des seuils.

## Résultats de cross-validation

Chaque cellule indique `moyenne ± écart-type` sur les cinq folds.

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 0,73465 ± 0,00009 | 0,00000 ± 0,00000 | 0,00000 ± 0,00000 | 0,00000 ± 0,00000 | 0,50000 ± 0,00000 |
| Logistic Regression | **0,80636 ± 0,00900** | **0,66942 ± 0,02050** | 0,53445 ± 0,03851 | **0,59365 ± 0,02622** | 0,84664 ± 0,01179 |
| Decision Tree | 0,72790 ± 0,00943 | 0,48697 ± 0,01820 | 0,48562 ± 0,03420 | 0,48598 ± 0,02489 | 0,65076 ± 0,01651 |
| Random Forest | 0,78719 ± 0,01279 | 0,62824 ± 0,03404 | 0,48696 ± 0,02214 | 0,54847 ± 0,02527 | 0,82369 ± 0,01129 |
| Gradient Boosting | 0,80245 ± 0,01095 | 0,65672 ± 0,02782 | **0,53645 ± 0,02284** | 0,59033 ± 0,02242 | **0,84789 ± 0,01198** |

Les vrais modèles dépassent la baseline sur les métriques utiles à la classe Churn. L'arbre non régularisé obtient même une accuracy légèrement inférieure au Dummy, mais détecte une partie des churners ; cela confirme qu'une seule métrique ne suffit pas.

## Classements

- **Accuracy :** Logistic Regression > Gradient Boosting > Random Forest > Dummy > Decision Tree.
- **Precision :** Logistic Regression > Gradient Boosting > Random Forest > Decision Tree > Dummy.
- **Recall :** Gradient Boosting > Logistic Regression > Random Forest > Decision Tree > Dummy.
- **F1 :** Logistic Regression > Gradient Boosting > Random Forest > Decision Tree > Dummy.
- **ROC-AUC :** Gradient Boosting > Logistic Regression > Random Forest > Decision Tree > Dummy.

Gradient Boosting est le candidat provisoire selon la règle ROC-AUC annoncée. Son avance sur Logistic Regression est faible (`0,00125`) et largement inférieure aux écarts-types des folds ; aucun modèle final n'est donc sélectionné.

![Comparaison CV](../reports/figures/ml_cv_model_comparison.png)

## Comparaison contrôlée du Feature Engineering

Le même Logistic Regression, les mêmes paramètres, les mêmes folds et les mêmes métriques sont utilisés.

| Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sans Feature Engineering | 0,80192 | 0,65209 | 0,54381 | 0,59241 | 0,84615 |
| Avec Feature Engineering | 0,80636 | 0,66942 | 0,53445 | 0,59365 | 0,84664 |
| Delta avec − sans | +0,00444 | +0,01733 | −0,00936 | +0,00124 | +0,00049 |

Le gain est faible et mixte : Precision augmente, Recall diminue, F1 et ROC-AUC changent très peu. Les features ne sont ni supprimées ni déclarées définitivement utiles ; leur sélection reste à confirmer dans les prochaines expériences.

## Matrices de confusion out-of-fold

Les prédictions proviennent de `cross_val_predict` sur le train : chaque ligne est prédite par un pipeline qui ne l'a pas utilisée pour son ajustement.

| Modèle | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 4 139 | 0 | 1 495 | 0 |
| Logistic Regression | 3 744 | 395 | 696 | 799 |
| Gradient Boosting | 3 719 | 420 | 693 | 802 |

![Matrices de confusion OOF](../reports/figures/ml_oof_confusion_matrices.png)

![Courbes ROC OOF](../reports/figures/ml_oof_roc_curves.png)

## Interprétation métier des erreurs

Un **False Negative** est un churner réel prédit `No Churn`. L'entreprise pourrait ne pas lui proposer d'action de rétention et manquer une opportunité d'éviter son départ. Logistic Regression en produit 696 en OOF et Gradient Boosting 693.

Un **False Positive** est un non-churner prédit `Churn`. L'entreprise pourrait engager inutilement une promotion ou une action commerciale. Logistic Regression en produit 395 et Gradient Boosting 420.

Au seuil par défaut, Gradient Boosting détecte trois churners supplémentaires, au prix de 25 fausses alertes supplémentaires par rapport à Logistic Regression. Le coût relatif de ces erreurs n'est pas connu. Recall pourrait devenir prioritaire si les opportunités manquées sont coûteuses, mais Precision reste importante si les actions de rétention sont onéreuses. Le compromis sera étudié par le threshold tuning à l'Étape 10.

## Déséquilibre de classes

Le train contient 73,46 % de `No` et 26,54 % de `Yes`. Le Dummy atteint ainsi 73,46 % d'accuracy tout en ayant un Recall Churn nul. Les modèles réels détectent environ 49–54 % des churners au seuil naturel, ce qui laisse encore près de la moitié des churners non détectés.

Aucune correction n'est appliquée : pas de class weights, SMOTE, sur-échantillonnage, sous-échantillonnage ou modification du seuil.

## Artefacts reproductibles

- `reports/ml_cv_results.csv` : résultats numériques CV.
- `reports/ml_evaluation.json` : protocole, résultats, comparaison FE et matrices OOF.
- trois figures sous `reports/figures/`.

Aucun pipeline ou modèle ajusté n'est sérialisé.
