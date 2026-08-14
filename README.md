# Customer Churn Prediction — End-to-End Data Science Project

Projet professionnel visant à construire progressivement un système complet de prédiction du churn client, depuis la compréhension du besoin métier jusqu'au déploiement.

## Objectif

Développer un pipeline reproductible permettant de préparer les données, entraîner et évaluer des modèles, expliquer leurs prédictions, puis exposer le résultat via des interfaces applicatives.

## Statut

🚧 **En cours de développement** — le système ML est figé, évalué, expliqué et exposé par FastAPI. Sa conteneurisation et sa configuration Render sont prêtes ; le déploiement public nécessite encore une action de compte et une validation Docker.

## Roadmap

Compréhension métier et des données → nettoyage et exploration → préparation et modélisation → évaluation et explicabilité → API et application → tests, conteneurisation et déploiement.

La formalisation métier et Data Science est disponible dans [la documentation de l'Étape 1](docs/business-understanding.md).

La provenance et l'inspection initiale du dataset IBM sont documentées dans [docs/dataset-provenance.md](docs/dataset-provenance.md) et [reports/data-understanding.md](reports/data-understanding.md).

Les règles et validations du nettoyage sont documentées dans [docs/data-cleaning.md](docs/data-cleaning.md).

L'analyse exploratoire détaillée se trouve dans [notebooks/01_exploratory_data_analysis.ipynb](notebooks/01_exploratory_data_analysis.ipynb), avec une synthèse dans [reports/eda-report.md](reports/eda-report.md).

Le protocole de préparation sans data leakage est décrit dans [docs/preprocessing.md](docs/preprocessing.md).

Les features déterministes et leur intégration au pipeline sont décrites dans [docs/feature-engineering.md](docs/feature-engineering.md).

La baseline, la comparaison des modèles et l'évaluation train-only sont documentées dans [docs/ml-evaluation.md](docs/ml-evaluation.md).

L'analyse du déséquilibre, le tuning, le choix de seuil et la sélection du candidat sont documentés dans [docs/model-improvement.md](docs/model-improvement.md).

L'évaluation finale, les coefficients, SHAP et la sérialisation du pipeline sont documentés dans [docs/final-ml-pipeline.md](docs/final-ml-pipeline.md). Sur le test jamais vu, le système obtient une ROC-AUC de 0,8429 et détecte 285 churners sur 374 au seuil figé de 0,30.

## Prérequis

- Python 3.11 ou version ultérieure

## API

Installation et lancement local :

```bash
python -m pip install -e .
uvicorn api.main:app --reload
```

Swagger : `http://127.0.0.1:8000/docs`.

Exemple minimal :

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @client.json
```

Réponse :

```json
{"churn_probability": 0.7878, "churn_prediction": 1, "churn_label": "Churn", "threshold": 0.3}
```

Le contrat complet, les 19 champs et les règles de validation sont documentés dans [docs/api.md](docs/api.md).

## User Interface

Une interface responsive destinée aux utilisateurs non techniques est servie directement par FastAPI :

```bash
uvicorn api.main:app --reload
```

Ouvrir `http://127.0.0.1:8000/`, charger l'exemple ou renseigner les 19 informations client, puis utiliser **Predict Churn Risk**. L'interface appelle le vrai endpoint `/predict` et affiche la probabilité, la classe et le seuil opérationnel de 30 %. Détails dans [docs/user-interface.md](docs/user-interface.md).

## Docker

```bash
docker build -t customer-churn-api:1.0.0 .
docker run --rm --name customer-churn-api -p 8000:8000 customer-churn-api:1.0.0
```

Swagger : `http://localhost:8000/docs`. Le conteneur accepte aussi un port fourni par `PORT` et exécute Uvicorn sans mode reload.

## Deployment

La cible préparée est un Web Service Docker Render configuré par [render.yaml](render.yaml). Statut actuel : **DEPLOYMENT-READY — manual account action required**. Aucune URL publique n'est annoncée tant qu'un compte Render n'a pas connecté et validé le dépôt.

L'image `customer-churn-api:1.0.0` a été réellement construite et validée avec Docker Desktop 29.7.2. Elle expose l'interface sur `/`, Swagger sur `/docs` et l'API sur les routes existantes. Le plan gratuit Render peut mettre le service en veille après une période d'inactivité ; la première requête suivante peut être plus lente. Les validations et le processus complet sont documentés dans [docs/docker-deployment.md](docs/docker-deployment.md).
