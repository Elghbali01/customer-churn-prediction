# Provenance du dataset

## Dataset retenu

- **Nom local :** `Telco-Customer-Churn.csv`
- **Variante :** table IBM classique à 21 colonnes et cible binaire `Churn`
- **Producteur :** IBM, données d'exemple relatives à une entreprise télécom fictive
- **URL d'acquisition :** `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Dépôt officiel :** `https://github.com/IBM/telco-customer-churn-on-icp4d`
- **Date d'acquisition du projet :** 2026-08-14
- **Taille :** 970 457 octets
- **SHA-256 :** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

IBM présente ce fichier dans son code pattern Cloud Pak for Data comme le dataset Telco Customer Churn à charger dans le projet. La documentation IBM Cognos décrit également le sample Telco customer churn comme les données d'une entreprise télécom fictive, avec une colonne de churn indiquant le départ du client.

## Justification de la variante

Cette variante est retenue parce qu'elle fournit une table client directement exploitable pour le futur problème de classification binaire, avec `customerID` comme identifiant candidat et `Churn` comme target. Elle est plus simple que la variante Cognos enrichie à cinq tables et ne contient pas les variables postérieures ou dérivées de la cible connues dans cette dernière.

## Acquisition reproductible et immuabilité

Le script `python -m customer_churn_prediction.acquire_data` télécharge le fichier depuis le dépôt officiel IBM, contrôle son SHA-256 et refuse d'écraser un fichier raw existant dont l'empreinte diffère. Le fichier sous `data/raw/` ne doit jamais être modifié ; toute transformation future devra produire un nouveau fichier sous `data/processed/`.

L'URL IBM historique `community.watsonanalytics.com/.../WA_Fn-UseC_-Telco-Customer-Churn.csv`, référencée par un ancien code pattern IBM, a d'abord été essayée mais n'était plus téléchargeable dans l'environnement. La copie du dépôt officiel IBM a donc été utilisée.

## Risques de target leakage

La variante enrichie IBM contient notamment `Customer Status`, `Churn Value`, `Churn Score`, `Churn Category` et `Churn Reason`. Ces champs révèlent directement la cible, en sont une représentation alternative, sont produits par un autre modèle ou ne sont disponibles qu'après le départ. Ils ne doivent pas être considérés automatiquement comme prédicteurs.

Ces colonnes sont absentes de la variante retenue. Cela ne dispense pas d'un audit temporel futur : chaque variable devra être disponible au moment réel où la prédiction serait effectuée.

## Références IBM

- IBM, dépôt officiel du code pattern : <https://github.com/IBM/telco-customer-churn-on-icp4d>
- IBM Community, description du sample Cognos enrichi : <https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113>
- IBM Documentation, sample Telco customer churn : <https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn>
