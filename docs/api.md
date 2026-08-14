# Étapes 13–15 — Architecture, FastAPI et tests

## Architecture

```text
Client JSON
→ FastAPI
→ validation Pydantic
→ predict_churn()
→ churn_pipeline.joblib
→ probabilité
→ seuil 0,30
→ réponse JSON validée
```

La logique ML reste dans `src/customer_churn_prediction/`. `api/` contient uniquement les contrats HTTP, le chargement centralisé des artefacts et les endpoints. Le pipeline et les métadonnées sont chargés une fois au démarrage depuis des chemins résolus relativement à la racine du projet, indépendamment du working directory.

## Installation et lancement

À partir de la racine du projet :

```bash
python -m pip install -e .
uvicorn api.main:app --reload
```

Swagger est disponible sur `http://127.0.0.1:8000/docs` et ReDoc sur `http://127.0.0.1:8000/redoc`.

## Endpoints

- `GET /` : identification minimale de l'API.
- `GET /health` : état réel du chargement du pipeline et version du modèle.
- `GET /model-info` : métadonnées non sensibles et métriques finales figées.
- `POST /predict` : prédiction d'un client.
- `POST /predict/batch` : prédiction ordonnée de 1 à 100 clients.

## Exemple de requête

```json
{
  "tenure": 5,
  "MonthlyCharges": 89.9,
  "TotalCharges": 450.5,
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
```

Réponse réelle avec l'artefact figé :

```json
{
  "churn_probability": 0.7877771751334054,
  "churn_prediction": 1,
  "churn_label": "Churn",
  "threshold": 0.3
}
```

## Validation

Le schéma accepte exactement les 19 features brutes. `customerID`, `Churn`, les features engineered et tout champ supplémentaire sont refusés. Les numériques doivent être non négatifs et finis ; les catégories sont limitées aux modalités historiques.

Deux règles inter-variables, directement supportées par le dataset, sont appliquées :

- sans service téléphonique, `MultipleLines` doit être `No phone service`; avec service, cette modalité est interdite ;
- sans service Internet, les six add-ons doivent être `No internet service`; avec Internet actif, cette modalité est interdite.

## Erreurs et confidentialité

- erreur de validation : HTTP 422 ;
- erreur réelle d'inférence : HTTP 500 sans résultat fictif ;
- artefact absent ou invalide : échec explicite au démarrage.

L'API ne stocke aucune requête, ne loggue pas les payloads complets, n'utilise ni base de données, ni authentification, ni secret hardcodé.
