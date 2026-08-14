# Data Dictionary — IBM Telco Customer Churn

Les types indiqués sont ceux observés lors de la lecture brute avec Pandas. Les rôles sont provisoires, sauf pour l'identifiant et la target. Aucun changement de type n'est appliqué ici.

| Variable | Signification documentée ou directement établie | Type observé | Rôle potentiel | Remarque observée |
| --- | --- | --- | --- | --- |
| `customerID` | Identifiant unique du client | `object` | Identifiant | 7 043 valeurs uniques ; à exclure probablement des prédicteurs directs, décision reportée |
| `gender` | Genre déclaré dans le sample (`Female`, `Male`) | `object` | Catégorielle | 2 modalités |
| `SeniorCitizen` | Indicateur de client senior | `int64` | Catégorielle binaire potentielle | Codé `0`/`1` ; signification exacte du seuil d'âge à confirmer pour cette variante |
| `Partner` | Indique si le client a un partenaire | `object` | Catégorielle | `Yes`/`No` |
| `Dependents` | Indique si le client a des personnes à charge | `object` | Catégorielle | `Yes`/`No` |
| `tenure` | Ancienneté du client, communément exprimée en mois dans ce sample | `int64` | Numérique | 73 valeurs distinctes ; unité à confirmer avec la documentation de la variante |
| `PhoneService` | Souscription au service téléphonique | `object` | Catégorielle | `Yes`/`No` |
| `MultipleLines` | Présence de plusieurs lignes téléphoniques | `object` | Catégorielle | Inclut `No phone service` |
| `InternetService` | Type de service Internet | `object` | Catégorielle | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | Souscription au service de sécurité en ligne | `object` | Catégorielle | Inclut `No internet service` |
| `OnlineBackup` | Souscription au service de sauvegarde en ligne | `object` | Catégorielle | Inclut `No internet service` |
| `DeviceProtection` | Souscription au service de protection des appareils | `object` | Catégorielle | Inclut `No internet service` |
| `TechSupport` | Souscription au support technique | `object` | Catégorielle | Inclut `No internet service` |
| `StreamingTV` | Souscription au service de télévision en streaming | `object` | Catégorielle | Inclut `No internet service` |
| `StreamingMovies` | Souscription au service de films en streaming | `object` | Catégorielle | Inclut `No internet service` |
| `Contract` | Type ou durée du contrat | `object` | Catégorielle | 3 modalités observées |
| `PaperlessBilling` | Utilisation de la facturation dématérialisée | `object` | Catégorielle | `Yes`/`No` |
| `PaymentMethod` | Méthode de paiement | `object` | Catégorielle | 4 modalités observées |
| `MonthlyCharges` | Montant facturé mensuellement | `float64` | Numérique | Aucune valeur manquante détectée par Pandas |
| `TotalCharges` | Montant total facturé | `object` | Numérique potentielle | 11 chaînes vides/espaces ; problème seulement détecté |
| `Churn` | Indique si le client a quitté l'entreprise | `object` | **Target** | `Yes` = classe positive, `No` = classe négative |

## Colonnes enrichies absentes et interdites par défaut

La variante retenue ne contient pas `Customer Status`, `Churn Value`, `Churn Score`, `Churn Category` ou `Churn Reason`. Si une future source les introduit, elles devront être interdites par défaut comme features jusqu'à preuve documentée de leur disponibilité antérieure à la cible et de l'absence de leakage.
