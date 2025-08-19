# Cat-Or-Not-IA

Petit projet pédagogique pour apprendre la classification d'images : entraîner un modèle (CNN) qui prédit si une image est un chat ou non.

## Aperçu rapide
- Les scripts principaux :
  - `data/prepare_data.py` : lit le CSV des images, ouvre/redimensionne/normalise, et sauvegarde `X.npy` et `y.npy`.
  - `train_keras.py` : charge `X.npy`/`y.npy`, entraîne un modèle Keras et sauvegarde le modèle.
  - `app.py` : interface Streamlit pour tester le modèle en local.
  - `run_all.sh` : pipeline simple qui exécute préparation → entraînement → streamlit.

## Prérequis
- Python 3.11.9 (le projet utilise précisément la version 3.11.9).
  - La version est déjà indiquée dans le fichier `.python-version` à la racine du projet.
- Utiliser un environnement virtuel (`.venv` recommandé).

Installation (bash)
```bash
python -m venv .venv
source .venv/Scripts/activate   # sous Windows Git Bash / WSL
pip install -r requirements.txt
```

Remarque : si tu utilises pyenv, tu peux choisir une version locale avant de créer le venv.

## Données attendues
Le script de préparation s'attend à trouver :
- Un CSV contenant deux colonnes : `image,labels` (ex : `cat_0001.jpg,1`).
  - Convention dans le projet : `labels` = 0 -> chat, 1 -> chien.
- Un dossier contenant les images.

Dans les scripts actuels les chemins par défaut sont :
- CSV : `data/raw_data/cat_dog.csv`
- Images : `data/raw_data/cat_dog/`

Si tu utilises un dataset, place les fichiers selon cette structure ou modifie `data/prepare_data.py` pour pointer vers tes chemins.

## Préparation des données
Le script centralise les paramètres dans `data/config.py` (par ex. `IMG_SIZE`, `SAMPLE_LIMIT`). Tu peux surcharger certains paramètres via variables d'environnement.

Exemples :
- Limiter pour tests rapides (bash) :
  ```bash
  LIMIT=500 python -m data.prepare_data
  ```
- Redéfinir la taille d'image :
  ```bash
  IMG_WIDTH=128 IMG_HEIGHT=128 python -m data.prepare_data
  ```

Important : exécuter `prepare_data` en mode package (`python -m data.prepare_data`) garantit que les imports internes `from data.config import ...` fonctionnent correctement.

À la fin, `prepare_data` écrit les sorties dans `data/transformed_data/` (par ex. `X.npy`, `y.npy`).

## Entraînement
Lancer l'entraînement depuis la racine du projet :
```bash
python -m train_keras
```

Que fait le script :
- Charge `data/transformed_data/X.npy` et `y.npy`.
- Sépare train/test, construit le modèle, entraîne et sauvegarde le modèle sous `data/transformed_data/cat_or_not_model.keras`.

Conseils rapides :
- Utilise `python -m train_keras` (exécution en mode module) pour que les imports absolus fonctionnent.
- Tu peux définir `IMG_WIDTH`/`IMG_HEIGHT` avant la préparation pour tester différentes tailles d'image.

## Lancer l'interface
```bash
streamlit run app.py
```

## Variables d'environnement utiles
- `LIMIT` : nombre maximal d'images à traiter (utile pour tests rapides).
- `IMG_WIDTH`, `IMG_HEIGHT` : redéfinissent `IMG_SIZE` sans modifier `data/config.py`.

Exemple complet (bash) :
```bash
IMG_WIDTH=128 IMG_HEIGHT=128 LIMIT=500 bash run_all.sh
```

## Bonnes pratiques & diagnostics rapides
- Toujours vérifier que `y` contient bien 0/1 et leur distribution :
  ```python
  import numpy as np
  y = np.load('data/transformed_data/y.npy')
  print(np.unique(y, return_counts=True))
  ```
- Si le modèle prédit toujours la même classe : calculer la matrice de confusion et l'histogramme des probabilités (voir `train_keras.py` pour ajouter ces diagnostics).
- Sauvegarde du meilleur modèle : utiliser `ModelCheckpoint` et `EarlyStopping` (réduction d'overfitting).

## Emplacements importants
- Données brutes (images + CSV) : `data/raw_data/` (par défaut)
- Données transformées (.npy) : `data/transformed_data/`
- Modèle sauvegardé : `data/transformed_data/cat_or_not_model.keras`

## Reprendre plus tard
Quand tu veux reprendre : active le venv, (ré)installe les dépendances si besoin, puis exécute `bash run_all.sh` ou les commandes séparées ci‑dessus.

Si tu veux, je peux appliquer des changements automatiques non invasifs maintenant (par ex. ajouter callbacks et diagnostics dans `train_keras.py`).