# Étape 1 — Business Understanding

## 1. Contexte métier

Dans un contexte télécom, le **churn** désigne le départ d'un client, par exemple lorsqu'il résilie son abonnement ou cesse d'utiliser les services de l'opérateur selon une définition qui devra être précisée avec les données disponibles.

Le churn constitue un problème métier car la perte de clients peut réduire les revenus futurs et conduire l'entreprise à engager des efforts pour remplacer les clients partis. À ce stade, l'impact économique exact n'est toutefois pas quantifié.

L'objectif général du projet est de développer un système de Machine Learning capable d'estimer le risque de churn d'un client télécom à partir de ses caractéristiques. Cette estimation doit aider l'entreprise à identifier les clients à risque et à prioriser ses actions de rétention. Elle constitue une aide à la décision : elle ne détermine pas, à elle seule, l'action commerciale à entreprendre.

## 2. Traduction en problème Data Science

Le problème est formulé comme un problème d'**apprentissage supervisé** et de **classification binaire** :

- classe positive : **Churn** ;
- classe négative : **No Churn** ;
- sortie souhaitée : une **probabilité de churn**, accompagnée d'une **décision de classification** obtenue à partir d'un seuil de décision.

Les variables explicatives ne sont pas fixées à ce stade. Elles devront être identifiées, évaluées et validées après la sélection et l'analyse du dataset, en veillant notamment à leur disponibilité au moment de la prédiction et aux risques de fuite de données.

## 3. Objectifs du système

Le futur système devra permettre de :

- identifier les clients présentant un risque de churn ;
- produire une probabilité de churn exploitable ;
- comprendre ultérieurement les principaux facteurs associés au churn ;
- fournir une information aidant à prioriser les actions de rétention.

## 4. Coût des erreurs

### Faux positif (False Positive)

Le système prédit qu'un client va churner alors que celui-ci serait resté. La conséquence potentielle est le déclenchement d'une action de rétention ou l'attribution d'une promotion inutile, avec le coût associé.

### Faux négatif (False Negative)

Le système prédit qu'un client restera alors qu'il churn réellement. La conséquence potentielle est une opportunité de rétention manquée et la perte du client.

Les faux négatifs peuvent être particulièrement importants : un client réellement à risque mais non détecté risque de ne recevoir aucune action préventive. Cela ne signifie pas qu'ils doivent être évités à tout prix, car une détection plus large peut également augmenter les fausses alertes. Aucun coût économique réel n'est actuellement disponible pour quantifier ou comparer précisément ces deux types d'erreurs.

## 5. Principes d'évaluation

L'accuracy seule sera insuffisante, notamment parce qu'elle ne décrit pas séparément la détection des churners et les erreurs commises sur chaque classe. Le Recall, la Precision, le F1-score, le ROC-AUC et la matrice de confusion seront étudiés lors de l'étape d'évaluation.

Le Recall de la classe Churn sera probablement important compte tenu du coût potentiel des faux négatifs. La métrique principale et le seuil de décision ne sont cependant pas encore choisis. Leur sélection devra être justifiée par le compromis métier entre le nombre de churners détectés et le nombre de fausses alertes, idéalement à partir d'informations économiques fiables.

Aucun objectif numérique de performance n'est fixé à ce stade.

## 6. Hypothèses et limites actuelles

- Aucun dataset n'est encore sélectionné définitivement.
- La définition opérationnelle exacte du churn dépendra du dataset et du contexte métier disponibles.
- Aucun coût économique réel n'est disponible.
- Aucune variable explicative n'est encore validée.
- Aucune métrique finale ni aucun seuil de décision ne sont encore choisis.
- Aucune performance Machine Learning ne peut encore être annoncée.
- La relation entre des caractéristiques et le churn ne devra pas être interprétée automatiquement comme une relation causale.

## 7. Questions laissées ouvertes

Les prochaines étapes devront notamment préciser la source et la qualité des données, la définition de la cible et de l'horizon de prédiction, la population concernée, le moment auquel une prédiction doit être produite, ainsi que les contraintes économiques et opérationnelles des actions de rétention.

