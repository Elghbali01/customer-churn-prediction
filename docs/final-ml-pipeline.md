# Étapes 11–12 — Explainability et pipeline ML final

## Protocole et évaluation finale

Le candidat sélectionné à l'Étape 10 a été figé avant toute observation du test : Feature Engineering activé, preprocessing existant, Logistic Regression (`C=2`, `solver=lbfgs`, pénalité L2, sans pondération) et seuil opérationnel `0,30`. Le pipeline a été ajusté une seule fois sur les 5 634 lignes du train. Le test de 1 409 lignes a ensuite été consommé pour la première évaluation finale, sans nouvelle sélection.

| Seuil | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,50 | 0,8048 | 0,6656 | 0,5321 | 0,5914 | 935 | 100 | 175 | 199 |
| **0,30** | **0,7608** | **0,5347** | **0,7620** | **0,6284** | **787** | **248** | **89** | **285** |

ROC-AUC test : `0,8429`. Average Precision test : `0,6379`. Le passage de 0,50 à 0,30 détecte 86 churners supplémentaires et évite 86 faux négatifs, contre 148 fausses alertes supplémentaires. Le seuil n'a pas été réoptimisé sur le test.

La ROC-AUC baisse de 0,0039 par rapport à sa moyenne CV (`0,8468`). L'Average Precision baisse de 0,0273 (`0,6652` vers `0,6379`). Au seuil 0,30, Precision passe de `0,5482` OOF à `0,5347` test, Recall de `0,7572` à `0,7620` et F1 de `0,6360` à `0,6284`. Ces écarts restent raisonnables et ne motivent aucune correction post-test.

## Feature importance par coefficients

Le preprocessing produit 72 colonnes : variables numériques standardisées et indicateurs One-Hot. Chaque nom est récupéré depuis le `ColumnTransformer` puis associé au coefficient correspondant. Un coefficient positif augmente le log-odds prédit de churn, toutes choses égales par ailleurs ; un coefficient négatif le réduit. `exp(coefficient)` est fourni comme odds ratio conditionnel au modèle.

Les plus grands coefficients absolus incluent `tenure`, plusieurs modalités de `contract_tenure`, `Contract_Two year`, `MonthlyCharges`, les modalités contractuelles mensuelles et les offres Internet. Leur signe ne doit pas être isolé de la représentation complète : variables sources et interactions sont corrélées, toutes les modalités One-Hot sont présentes, et les variables numériques sont standardisées. Certains signes contre-intuitifs illustrent précisément cette multicolinéarité. **Association prédictive ≠ causalité.**

Le tableau exhaustif est disponible dans `reports/final_model_coefficients.csv`.

## SHAP

`shap.LinearExplainer` explique exactement la Logistic Regression contenue dans le pipeline final, à partir des données transformées par le Feature Engineering, le scaling et le One-Hot Encoding. Le mapping a été validé : `1 409 × 72` valeurs SHAP finies pour `72` noms transformés.

L'importance globale moyenne absolue place notamment en tête `tenure`, `MonthlyCharges`, les modalités de contrat, `TotalCharges`, le service Internet et plusieurs interactions engineered. `total_services` apparaît au 11e rang global : son information est visible, sans dominer le modèle.

L'exemple individuel est choisi de façon reproductible comme la plus forte probabilité du test (`0,8860`). Le modèle le classe churner au seuil 0,30 et sa vraie classe est Churn. Les contributions positives principales incluent une faible ancienneté représentée par `tenure`, contrat mensuel, fibre et leurs interactions. Les contributions négatives principales concernent notamment `MonthlyCharges`, `TotalCharges` et certaines modalités d'ancienneté absentes ou centrées. Cette explication illustre une prédiction ; elle ne sert pas à sélectionner le client, le modèle ou le seuil.

## Interprétation métier prudente

Le modèle associe davantage de risque prédit aux relations récentes, aux contrats mensuels et à certains profils fibre/mensuel. Les contrats longs et une ancienneté élevée apparaissent globalement associés à un risque plus faible, malgré des coefficients individuels parfois contre-intuitifs en présence d'interactions redondantes. Ces tendances sont cohérentes avec l'EDA au niveau des grands profils, sans démontrer de causalité.

Une équipe de rétention pourrait tester des campagnes ciblant les clients récents à contrat mensuel, segmenter les actions selon l'offre Internet et évaluer expérimentalement des incitations à l'engagement ou à l'équipement. Le modèle ne prouve pas qu'une intervention réduira le churn ; les actions doivent faire l'objet de tests contrôlés et d'une analyse économique.

## Pipeline final et sérialisation

```text
features brutes
→ ChurnFeatureEngineer
→ ColumnTransformer (StandardScaler + OneHotEncoder)
→ LogisticRegression
→ probabilité
→ seuil explicite 0,30
→ classe opérationnelle
```

Le pipeline complet entraîné est sauvegardé dans `models/churn_pipeline.joblib`. Le seuil, les paramètres, le nombre de features, le périmètre d'entraînement et les métriques test sont stockés dans `models/model_metadata.json`. Une validation save/load sur 20 observations donne une différence maximale de probabilité de `0,0` et des classes identiques.

La fonction `predict_churn` accepte les 19 features brutes et retourne `churn_probability`, `churn_prediction` et `threshold`. Elle prépare l'intégration future, sans créer d'API.

## Limites

- Un seul dataset télécom historique est utilisé.
- Aucune calibration économique du seuil n'est disponible.
- Les coefficients et SHAP expliquent le comportement du modèle, pas les causes du churn.
- Les variables corrélées et interactions peuvent se partager ou déplacer l'importance.
- SHAP opère dans l'espace transformé ; une même variable métier peut apparaître sous plusieurs modalités.
- Le pipeline devra être surveillé face aux changements de population, catégories et pratiques commerciales.
