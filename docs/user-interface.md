# User Interface

## Choix technologique

L'interface utilise HTML, CSS et JavaScript natifs servis directement par FastAPI. Ce choix conserve une seule application et un seul déploiement, sans Node.js, React, Streamlit, moteur de templates ou dépendance frontend. FastAPI reste le backend réel et le navigateur appelle exclusivement `POST /predict`.

## Architecture

```text
Utilisateur
→ interface responsive GET /
→ validation HTML et comportements JavaScript
→ POST /predict
→ validation Pydantic
→ predict_churn()
→ pipeline sérialisé
→ probabilité et classe au seuil 0,30
→ présentation du résultat
```

Les fichiers sont organisés sous `api/static/` : `index.html`, `css/app.css` et `js/app.js`. La documentation Swagger reste disponible sur `/docs`.

## Formulaire

Les 19 features brutes sont réparties en cinq sections :

1. profil client : genre, senior, partenaire, personnes à charge ;
2. compte : ancienneté, contrat, facturation dématérialisée ;
3. téléphonie : service et lignes multiples ;
4. Internet : technologie et six services ;
5. facturation : charges mensuelles, charges totales et paiement.

`customerID`, `Churn` et les quatre features engineered ne sont jamais demandés. Les nombres utilisent des contrôles numériques non négatifs et les catégories des listes fermées.

## Cohérence dynamique

- Lorsque le service téléphonique est absent, `MultipleLines` devient `No phone service` et est désactivé.
- Lorsque le service Internet est absent, les six add-ons deviennent `No internet service` et sont désactivés.
- Le backend Pydantic reste la source de vérité et revalide toujours le payload.

## Exemple et prédiction

`Load Example` renseigne le même client fibre/mensuel que la documentation API : tenure 5, charges mensuelles 89,90 et total 450,50. Le résultat réel est une probabilité `0,7877771751334054`, la classe `Churn` et le seuil `0,30`.

Pendant l'appel, le bouton est désactivé et affiche `Predicting…`. Le résultat présente une jauge CSS, la probabilité, la classe, le seuil et le code modèle. Aucun calcul ML n'est effectué dans le navigateur.

## Niveau de risque

Les libellés visuels sont uniquement des catégories UX :

- Low : probabilité `< 0,30` ;
- Moderate : `0,30 ≤ probabilité < 0,60` ;
- High : probabilité `≥ 0,60`.

Ils sont affichés avec le suffixe `(presentation)` et ne modifient jamais le seuil ML. La règle opérationnelle reste : probabilité `≥ 0,30` → `Churn`.

## Erreurs, accessibilité et responsive

L'interface distingue validation 422, erreur serveur et indisponibilité réseau, sans afficher de stack trace. Elle vérifie `/health` au chargement. Les labels sont associés aux contrôles, les statuts utilisent du texte en plus de la couleur, les focus clavier sont visibles et le mouvement réduit est respecté.

Le layout passe de deux colonnes à une colonne, puis transforme toutes les grilles de champs en une colonne sous 680 px. Une validation réelle à 390 × 844 a confirmé l'absence de débordement horizontal.

## Lancement

```bash
uvicorn api.main:app --reload
```

Puis ouvrir `http://127.0.0.1:8000/`. En Docker : `http://localhost:8000/`.

## Limites

L'interface n'effectue pas d'explication individuelle SHAP et ne mémorise aucun client. Les catégories Low/Moderate/High sont des aides visuelles non entraînées. Aucun test navigateur lourd n'est ajouté ; les routes/assets sont testés automatiquement et le workflow complet a été validé dans le navigateur intégré.
