import os
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import pil_to_array

# Paramètres
IMG_SIZE = (64, 64)  # Taille à laquelle on redimensionne toutes les images
CSV_PATH = "data/cat_dog.csv"  # Chemin du fichier CSV
IMG_DIR = "data/cat_dog"  # Dossier contenant les images
# Permet de limiter le nombre d'images pour des tests rapides (mettre None pour tout traiter)
LIMIT = int(os.getenv("LIMIT", "0")) or None

# 1. Lire le CSV
# On utilise pandas pour charger le fichier CSV qui contient les chemins des images et leur label (cat ou dog)
df = pd.read_csv(CSV_PATH)

if LIMIT is not None:
    df = df.head(LIMIT)  # Pour tester, on ne garde que les premières lignes

# 2. Préparer les listes pour stocker les images et les labels
X = []  # Images
y = []  # Labels (0 = dog, 1 = cat)

# 3. Parcourir chaque ligne du CSV

# .iterrows() permet de parcourir les lignes d'un DataFrame
# idx: l'index de la ligne => int
# row: une Series contenant les données de la ligne (index: ['image', 'labels'])
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Traitement des images"):
    img_filename = row["image"]  # Nom du fichier image
    label = row["labels"]  # Label (0 = dog, 1 = cat)
    img_path = osp.join(IMG_DIR, img_filename)
    try:
        # a. Ouvrir l'image

        # Utiliser la fonction partagée pil_to_array pour ouvrir, redimensionner et normaliser
        pil = Image.open(img_path)
        img_array = pil_to_array(pil, size=IMG_SIZE)
        X.append(img_array)

        # d. Ajouter le label (déjà 0 ou 1)
        y.append(label)
    except Exception as e:
        print(f"Erreur avec l'image {img_path}: {e}")

# 4. Transformer les listes en tableaux numpy
X = np.array(X)
y = np.array(y)

print(f"Nombre d'images chargées: {len(X)}")
# .shape donne la forme du tableau, ici d'un élément de X (hauteur, largeur, canaux)
print(f"Taille des images: {X.shape[1:]} (doit être {IMG_SIZE} + (3,))")
# np.unique retourne 2 tableaux: les valeurs uniques et leur nombre d'occurrences
# le premier tableau contient les labels uniques (par ex [0, 1])
# le deuxième tableau contient le nombre d'occurrences de chaque label (par ex [1500, 1500])
print(f"Exemple de labels: {np.unique(y, return_counts=True)}")


def save_arrays(x_out="data/X.npy", y_out="data/y.npy"):
    """Sauvegarde les tableaux X et y dans des fichiers .npy.
    
    Args:
        x_out (str): Chemin du fichier de sortie pour les images.
        y_out (str): Chemin du fichier de sortie pour les labels.
    
    Returns:
        None
    """
    # les fichiers .npy sont des fichiers binaires optimisés pour stocker des tableaux numpy
    # Crée le dossier si nécessaire
    out_dir = osp.dirname(x_out)
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    np.save(x_out, X)
    np.save(y_out, y)
    print(f"Données sauvegardées dans {x_out} et {y_out}")


# on passe par une liste avant de convertir en tableau numpy car un tableau a une forme fixe, or ici on ne connaît pas le nombre d'images

if __name__ == "__main__":
    # Permet d'exécuter le script directement
    save_arrays()
