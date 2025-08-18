from PIL import Image
import numpy as np

# Fonction utilitaire de prétraitement d'image qui reproduit exactement la logique de prepare_data.py
# (conversion en RGB, redimensionnement, puis division par 255.0)


def pil_to_array(pil_img, size=(64, 64)):
    """Convertit une PIL.Image en tableau numpy normalisé (H, W, 3).

    Le comportement suit prepare_data.py :
    - pil_img.convert("RGB")
    - pil_img.resize(size)
    - np.array(img) / 255.0

    Args:
        pil_img (PIL.Image): Image à convertir.

    Returns:
        np.ndarray: Tableau numpy de forme (H, W, 3) avec des valeurs entre 0 et 1.
    """
    # Convertir en RGB pour uniformiser les canaux de couleur
    # retourne un objet PIL Image, conçu pour être facilement manipulé
    img = pil_img.convert("RGB")

    # Redimensionner
    img = img.resize(size)

    # On évite d'imposer astype("float32") ici pour rester identique à prepare_data.py
    # transforme l'image en tableau numpy et normalise les valeurs entre 0 et 1 (les valeurs de pixel sont entre 0 et 255)
    # donne un tableau de 4096 (64x64) triplets de valeurs entre 0 et 1
    arr = np.array(img) / 255.0

    return arr


def load_and_preprocess(path, size=(64, 64)):
    """Ouvre une image depuis disque et la prétraite en utilisant pil_to_array.

    Args:
        path: chemin vers le fichier image
        size: taille de redimensionnement

    Returns:
        np.ndarray: Tableau numpy de l'image prétraitée.
    """
    pil = Image.open(path)
    return pil_to_array(pil, size)
