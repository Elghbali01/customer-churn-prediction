# Étapes 16–17 — Docker et préparation du déploiement

## État initial

Avant modification, l'API locale passait 38 tests et 13 subtests. Le pipeline sérialisé avait pour SHA-256 `1E90B4D71CB43D0778C816A7101836649EF53E92597FEF5AB5BA05EA2DA86B40`. L'environnement global contenait un conflit entre les besoins scikit-learn de plusieurs projets ; le conteneur isole désormais les dépendances déclarées par ce projet.

Docker Desktop 29.7.2 a ensuite été installé et validé. L'image finale a été réellement construite et exécutée, puis tous les endpoints, l'interface, les validations 422, le healthcheck et le hash intra-conteneur ont été contrôlés. Le statut cloud reste **DEPLOYMENT-READY — manual account action required**, car aucun compte Render n'est connecté.

## Architecture Docker

```text
Image python:3.11-slim-bookworm
→ installation du projet depuis pyproject.toml
→ API + artefacts figés
→ utilisateur non-root `app`
→ Uvicorn sur 0.0.0.0:${PORT:-8000}
→ FastAPI
→ pipeline sérialisé
```

Le build reste volontairement mono-stage : le projet Python ne compile pas d'artefact applicatif distinct et un multi-stage ajouterait de la complexité sans gain clair. Docker Compose n'est pas créé, car l'architecture ne contient qu'un service, sans base de données, cache ou queue.

## Build

```bash
docker build -t customer-churn-api:1.0.0 .
```

Vérification optionnelle des versions isolées :

```bash
docker run --rm customer-churn-api:1.0.0 python -c "import sys, fastapi, pydantic, sklearn, shap; print(sys.version); print(fastapi.__version__, pydantic.__version__, sklearn.__version__, shap.__version__)"
```

## Run

```bash
docker run --rm --name customer-churn-api -p 8000:8000 customer-churn-api:1.0.0
```

Pour tester un port fourni par l'environnement :

```bash
docker run --rm --name customer-churn-api -e PORT=8080 -p 8080:8080 customer-churn-api:1.0.0
```

Swagger est alors disponible sur `http://localhost:8000/docs`.

## Healthcheck et artefact

Le `HEALTHCHECK` utilise uniquement `urllib.request` de la bibliothèque standard et interroge `/health`. Aucun `curl` ni paquet système n'est installé.

Vérification du modèle dans le conteneur :

```bash
docker run --rm customer-churn-api:1.0.0 sha256sum /app/models/churn_pipeline.joblib
```

Valeur attendue :

```text
1E90B4D71CB43D0778C816A7101836649EF53E92597FEF5AB5BA05EA2DA86B40
```

## Validation d'intégration manuelle

Après démarrage, vérifier :

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl http://localhost:8000/docs
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @client.json
curl -X POST http://localhost:8000/predict/batch -H "Content-Type: application/json" -d @batch.json
```

Tester aussi `tenure=-1` et une catégorie inexistante ; les deux réponses doivent avoir le statut HTTP 422. Comparer la probabilité de l'exemple de référence à `0.7877771751334054`.

## Configuration de production

- Uvicorn écoute sur `0.0.0.0`.
- `PORT` est fourni par l'environnement, avec repli local sur 8000.
- `--reload` est absent.
- Un seul worker est utilisé pour limiter la mémoire du modèle sur une petite instance.
- Les access logs sont désactivés afin de réduire le bruit et l'exposition des chemins ; aucun payload n'est loggué.
- Le processus s'exécute avec l'utilisateur système non-root `app`.
- Aucun secret ou fichier `.env` n'entre dans l'image.

## Plateforme retenue : Render

Render est retenu pour ce projet portfolio : build Docker depuis le dépôt, URL HTTPS gérée, healthcheck HTTP et plan Web Service gratuit. La configuration unique se trouve dans `render.yaml`.

Processus manuel :

1. publier le repository, y compris les deux artefacts explicitement autorisés dans `.gitignore` ;
2. créer ou connecter un compte Render et le fournisseur Git ;
3. créer un Blueprint à partir de `render.yaml` ;
4. vérifier le build, `/health`, `/model-info`, `/predict`, `/predict/batch` et `/docs` ;
5. enregistrer l'URL publique réelle dans le README seulement après validation.

Le plan gratuit Render met actuellement les services en veille après 15 minutes sans trafic ; la première requête après inactivité peut prendre environ une minute. Cette limitation convient à une démonstration portfolio, pas à un SLA de production.

## Validation réalisée

- Image finale : `customer-churn-api:1.0.0`.
- Docker Desktop : 29.7.2, moteur Linux x86_64.
- Taille : environ 264,1 MB.
- Conteneur `healthy`, exécuté avec `uid=100(app)`.
- `/`, assets, `/health`, `/model-info`, `/predict`, `/predict/batch` et `/docs` : HTTP 200.
- Trois erreurs testées : numérique négatif, catégorie inconnue et incohérence métier → HTTP 422.
- Probabilité locale/Docker identique, différence maximale `0,0`.
- SHA-256 repository/conteneur identique.
- Tous les conteneurs temporaires ont été supprimés.

## Limites

- Aucun compte ou jeton Render n'est configuré : aucune URL publique n'a été créée.
- L'image inclut SHAP et les dépendances scientifiques déclarées, ce qui porte sa taille à environ 264,1 MB même si l'endpoint d'inférence n'exécute pas SHAP.
- Le plan Render gratuit est limité et peut subir un cold start.
- CI/CD, monitoring, drift detection et retraining restent hors périmètre.
