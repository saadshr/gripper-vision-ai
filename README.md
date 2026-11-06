# Gripper Vision AI — Eurostyle Systems Tanger

Prototype pédagogique : proposer un type de préhenseur à partir d'une photo de pièce. Une seule interface Streamlit, un modèle RandomForest historique et un pipeline OpenCV partagé entre extraction et prédiction. La masse, la matière et les contraintes physiques ne sont pas encore des entrées du modèle. Aucune commande de robot.

## Architecture

```text
GRIPPER_VISION_FINAL/
├── app.py                        seule interface Streamlit
├── src/
│   ├── config.py                 chemins et constantes
│   ├── safety.py                 bloque préparation et entraînement
│   ├── dataset.py                extraction des features (conservée, bloquée)
│   ├── cv/
│   │   ├── segmentation.py       image BGR → contour
│   │   ├── features.py           contour → 26 nombres
│   │   ├── visualize.py          contour et boîte superposés à l'image
│   │   └── augmentation.py       transformations conservées, non utilisées par l'app
│   ├── prediction/predictor.py   chargement joblib et prédiction
│   ├── ml/models.py              registre de modèles, XGBoost optionnel
│   ├── ml/evaluation.py          métriques et graphiques d'expériences
│   ├── training/trainer.py       découpage par source, CV et sélection
│   └── utils/logger.py           logger sans fichier créé automatiquement
├── data/
│   ├── photos/<classe>/          125 photos, noms et classes conservés
│   ├── a_verifier/               8 images exclues du jeu actif
│   ├── annotations.csv           labels hérités, SHA-256, provenance et statut
│   ├── quasi_doublons_a_verifier.csv
│   └── README.md
├── models/
│   ├── best_model.joblib         seul modèle utilisé
│   ├── label_encoder.joblib      correspondance indices/classes
│   └── model_metadata.json       métadonnées historiques exactes
├── tests/
│   ├── test_features.py          segmentation et géométrie
│   ├── test_training_split.py    absence de recouvrement entre groupes
│   ├── test_prediction.py        vraie inférence et garde d'entraînement
│   └── test_app.py               exécution Streamlit via AppTest
├── results/
│   ├── history/                  CSV de features, comparaison et rapport exacts
│   ├── provenance.json           paramètres historiques et sources
│   ├── dependencies_before.csv   graphe des imports avant simplification
│   ├── exclusions.csv            éléments non repris et justification
│   ├── files_retained.csv        fichiers retenus, provenance, empreintes
│   ├── protected_sources_state.csv
│   ├── consolidation_fingerprints.csv
│   └── vérifications finales     générées après les tests, sans métrique ML
├── requirements.txt
├── requirements-optional.txt
└── README.md
```

Les __init__.py rendent les sous-dossiers importables. REFERENCES_STAGE, à côté, contient le template ENSAM complet, le rapport PFE ENSAM, le rapport Eurostyle gestion et les logos. Les anciens rapports Gripper et les autres modèles restent dans STAGE_EUROSTYLE_FINAL et les archives originales. Rien de ces deux ensembles n'est modifié.

## Comprendre le flux

1. app.py reçoit les octets d'une photo importée, ou lit une photo historique choisie explicitement. Aucun upload temporaire n'est créé.
2. OpenCV décode l'image couleur BGR. segmentation.py conserve les proportions, passe en niveaux de gris, applique flou, Otsu et morphologie, puis choisit le plus grand contour.
3. features.py calcule 26 caractéristiques : aire, périmètre, dimensions en pixels, circularité, solidité, trous, coins, centroïde et moments de Hu.
4. predictor.py charge une fois le modèle et l'encodeur. Il remet les colonnes dans l'ordre du modèle, appelle predict et predict_proba, sans fit.
5. L'interface affiche la classe proposée, les scores et la segmentation à contrôler visuellement. Un score n'est pas une garantie de succès mécanique.

config.py calcule tous les chemins à partir de son emplacement, pas du terminal courant. dataset.py conserve l'extraction multi-format et la provenance des variantes. trainer.py conserve la logique historique de groupes, CV et sélection ; models.py fournit les candidats et evaluation.py les métriques. Ces opérations restent bloquées par safety.py tant que l'entraînement et la préparation ne sont pas explicitement autorisés. Le générateur synthétique et la GUI bureau ne sont plus des dépendances actives.

## Installation sous Windows — à exécuter seulement après autorisation

Python 3.12 est la version de référence. Aucune installation n'a été effectuée pendant cette simplification. Le modèle contient scikit-learn **1.8.0**, alors que l'ancien environnement contient **1.5.1** : les prédictions de contrôle s'exécutent avec un avertissement, sans garantie de compatibilité numérique. Le fichier requirements cible 1.8.0 pour correspondre aux poids ; cette installation cible reste à tester après autorisation. Depuis le dossier du projet, dans PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

L'appel direct à python.exe évite de modifier la politique d'exécution PowerShell. À l'exception de scikit-learn, les versions principales correspondent à l'environnement déjà présent utilisé pour les contrôles ; matplotlib 3.8.4 remplace la déclaration historique 3.9.1, tandis que les versions d'origine restent enregistrées dans results/provenance.json. L'installation fraîche et les dépendances transitives n'ont pas été vérifiées ; les versions principales figées ne constituent pas un verrou complet. En cas d'erreur Windows « nom de fichier ou extension trop long », prévoir un environnement situé dans un chemin court après autorisation, sans déplacer les sources protégées.

CustomTkinter et Tkinter étaient utilisés uniquement par l'interface bureau exclue. XGBoost n'est pas nécessaire à l'inférence du RandomForest : son import optionnel est conservé dans models.py et sa version dans requirements-optional.txt. Il est absent de l'environnement de contrôle ; aucune suppression de son code et aucune installation automatique.

## Utiliser et tester

Dans Prédiction, importer une seule pièce sur fond contrasté, ou choisir Essayer une photo du dataset puis Analyser la photo. Les photos du dataset ont déjà été vues lors de l'ajustement final historique : ce choix montre le fonctionnement logiciel et ne mesure pas la généralisation.

Les pages Données et résultats et À propos expliquent les données et limites. Aucun bouton d'entraînement. Aucun dossier n'est créé au démarrage de l'app.

Après installation autorisée, les tests se lancent sans entraînement :

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Lire results/VERIFICATIONS.md pour le résultat réel des contrôles de cette livraison. AppTest exécute le script Streamlit et une prédiction ; ce n'est pas une vérification visuelle dans un navigateur, ni un test robot. Les tests géométriques dessinent uniquement des formes en mémoire : aucun dataset synthétique n'est ajouté.

## Expériences et limites actuelles

- Les fichiers de results/history et models sont copiés sans modifier leurs octets. RandomForest sélectionné historiquement : test 0.32, CV 0.33, F1 macro environ 0.325. Ce ne sont pas de nouveaux résultats.
- Le DecisionTree à 0.52 dans l'ancien tableau n'est pas le modèle sélectionné. Les 100 % des anciens jeux synthétiques ne sont pas repris comme performance industrielle.
- Le modèle sauvegardé a ensuite été ajusté sur toutes les données ; les anciennes photos test ne sont plus indépendantes de ses poids.
- Le CSV historique comporte 625 lignes mais les 500 images augmentées manquent. Les paramètres, provenance et versions nécessaires à une tentative de reproduction sont conservés ; une reproduction bit à bit n'est pas garantie.
- Le modèle est sérialisé avec scikit-learn 1.8.0 mais l'environnement de test contient 1.5.1. L'application affiche cet écart lorsqu'il est détecté ; les avertissements de la bibliothèque restent visibles. Une inférence qui s'exécute ne prouve pas la validité des valeurs. Tester l'environnement cible 1.8.0 sur autorisation, sans réentraîner ni modifier les poids.
- Matplotlib 3.8.4 ne charge pas sa DLL dans le long chemin du venv historique. Les imports graphiques sont différés : métriques, split et prédiction restent accessibles ; les fonctions de figures sont conservées mais non validées dans cet environnement. Aucun code graphique n'a été supprimé pour masquer ce problème.
- La CV historique utilise des groupes par nom de photo et inclut des variantes en validation. Le repli pour un vieux CSV sans source_image est incomplet. Ces problèmes sont conservés et documentés, pas masqués par une réécriture algorithmique.
- Les noms sans extension peuvent se rencontrer ; les identifiants de pièces physiques restent à établir. Les labels sont hérités, pas validés par un expert atelier. Aucune fusion de labels contradictoires.
- Otsu et le plus grand contour peuvent échouer avec reflets, transparence, fonds complexes ou scènes multiples. Le comptage de trous porte sur toute la hiérarchie. Les dimensions sont des pixels sans calibration physique.
- Les probabilités ne sont pas calibrées ; le seuil 60 % est indicatif. Plusieurs préhenseurs peuvent convenir à une même pièce.
- Aucun nouveau split n'est fabriqué. Il faudra isoler un test par pièce/famille avant toute augmentation et tout choix de modèle.
- Toute future préparation devra créer une expérience dans un nouveau répertoire, sans écraser models/ ni les résultats historiques. Le retrait de la garde nécessite d'abord une nouvelle autorisation.

Les tests d'exécution n'affirment pas une qualité industrielle ou une accuracy améliorée. Aucun entraînement ni rédaction du rapport de stage pendant cette phase.
