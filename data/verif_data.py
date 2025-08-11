import numpy as np


def load_and_verify_data(x_path="data/X.npy", y_path="data/y.npy"):
    try:
        # Charger les données
        X = np.load(x_path)
        y = np.load(y_path)

        # Vérifications basiques
        print(f"Nombre d'images: {len(X)}")
        # Vérification des dimensions
        if len(X) != len(y):
            print("Erreur: Le nombre d'images et de labels ne correspond pas!")
        else:
            print("Les dimensions des données sont correctes.")

        print(f"Taille des images: {X.shape[1:]}")
        # Vérification du format des images
        if X.shape[1:] == (64, 64, 3):
            print("Format des images: Correct")
        else:
            print("Erreur: Format des images incorrect. Taille attendu: (64, 64, 3)")

        # Afficher les labels uniques et leur distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        print("Distribution des labels:")
        for label, count in zip(unique_labels, counts):
            print(f"  Label {label}: {count} occurrences")

    except Exception as e:
        print(f"Erreur lors du chargement ou de la vérification des données: {e}")


if __name__ == "__main__":
    load_and_verify_data()
