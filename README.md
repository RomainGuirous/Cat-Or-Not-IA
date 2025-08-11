# Cat-Or-Not-IA

Petit projet pédagogique: entraîner un modèle simple (CNN) pour reconnaître si une image est un chat ou non.

## Prérequis
- Python géré avec pyenv (ex: 3.11.9 via `pyenv local 3.11.9`)
- Environnement virtuel activé (`.venv` recommandé)

## Installation rapide

```bash
# 1) (optionnel) forcer la version locale de Python
pyenv install -s 3.11.9
pyenv local 3.11.9

# 2) créer et activer un venv
python -m venv .venv
source .venv/Scripts/activate

# 3) installer les dépendances
pip install -r requirements.txt
```

## Données

Téléchargement via `kagglehub`:

```bash
python data/download_images.py
```

Le chemin de téléchargement s'affiche en sortie. Copiez/organisez les fichiers attendus par `prepare_data.py`:
- `data/cat_dog/` dossier avec les images
- `data/cat_dog.csv` CSV avec colonnes `image,labels` (labels: 0 = dog, 1 = cat)

## Préparation des données

```bash
# Limiter le nombre d'images pour un test rapide (ex: 500)
LIMIT=500 python data/prepare_data.py

# Sans limite
python data/prepare_data.py
```

Ce script génère `data/X.npy` et `data/y.npy`.

## Entraînement

```bash
python train_keras.py
```

Le modèle entraîné est sauvegardé sous `cat_or_not_model.keras`.

## Notes
- Code simple et commenté pour apprendre les bases.
- Ajustez `IMG_SIZE` et `epochs` selon votre machine et le temps disponible.