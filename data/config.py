import os
# Configuration centrale pour le traitement des images et des chemins
# Modifiez IMG_SIZE ici pour propager le changement dans tout le projet
# Conventions :
# - IMG_SIZE est le tuple (largeur, hauteur) utilisé par PIL.Image.resize
# - CHANNELS est le nombre de canaux de couleur (généralement 3 pour RGB)
# - IMG_SHAPE est le tuple (hauteur, largeur, canaux) adapté à l'`input_shape` des modèles et aux tableaux numpy

DEFAULT_IMG_SIZE = (128, 128)  # (largeur, hauteur) pour PIL.resize

_env_img_width = os.getenv("IMG_WIDTH")
_env_img_height = os.getenv("IMG_HEIGHT")

if _env_img_width is not None and _env_img_height is not None:
    try:
        IMG_SIZE = (int(_env_img_width), int(_env_img_height))
    except Exception:
        IMG_SIZE = DEFAULT_IMG_SIZE
else:
    IMG_SIZE = DEFAULT_IMG_SIZE
# commande à rentrer dans le terminal pour définir IMG_SIZE :
# IMG_WIDTH=128 IMG_HEIGHT=128 run_all.sh


CHANNELS = 3
# Forme dérivée pour les tableaux numpy / entrée du modèle : (hauteur, largeur, canaux)
IMG_SHAPE = (IMG_SIZE[1], IMG_SIZE[0], CHANNELS)

# Taille d'échantillon pour tests rapides :
# - mettre None pour traiter tout le dataset
# - mettre un entier (ex: 500) pour limiter le nombre d'images traitées
# Cette valeur peut être surchargée par la variable d'environnement LIMIT si définie.
SAMPLE_LIMIT = None
