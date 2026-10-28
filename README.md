# FootPredict AI — prototype web gratuit

Application Streamlit pour calculer des probabilités de buts à partir d'un historique de résultats fourni en CSV.

## Fonctions
- Import CSV d'historiques de matchs.
- Estimation de buts attendus à domicile et à l'extérieur.
- Probabilités de moins/plus de 1,5, 2,5, 3,5, 4,5 et 5,5 buts.
- Probabilités 1N2 et scores exacts les plus probables.
- Affichage du volume de données et des limites du modèle.

## Important
- `data/exemple_fictif.csv` contient des données inventées pour démontrer l'interface. Elles ne doivent pas servir à analyser de vrais matchs.
- Le prototype n'intègre pas encore de flux de données automatique, de blessures, de compositions, ni de modèle calibré.
- Les résultats dépendent de la qualité, de la fraîcheur et de la couverture des données importées.
- Le moteur actuel est une base Poisson indépendante et doit être évalué sur des données historiques avant tout usage sérieux.

## Lancer sur son ordinateur
Python 3.11 recommandé.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Le navigateur ouvrira une adresse locale, généralement `http://localhost:8501`.

## Hébergement gratuit via Streamlit Community Cloud
1. Crée un compte GitHub et un dépôt public.
2. Téléverse `app.py`, `requirements.txt`, `data/exemple_fictif.csv` et ce README dans le dépôt.
3. Connecte-toi à Streamlit Community Cloud : https://share.streamlit.io/
4. Choisis le dépôt, la branche et le fichier `app.py`.
5. Clique sur Deploy.

Les services gratuits peuvent modifier leurs limites et conditions. L'application ne contient pas de clé API ni de secret. Pour une utilisation plus large, vérifier les règles de confidentialité et les conditions du service d'hébergement.

## Format CSV requis
En-têtes exacts :
`date,competition,home_team,away_team,home_goals,away_goals`

Exemple :
`2026-09-01,Ligue Exemple,Équipe A,Équipe B,2,1`

Chaque ligne représente un match terminé. Utiliser des noms d'équipe cohérents. Éviter les doublons et mélanges de compétitions/niveaux non comparables.
