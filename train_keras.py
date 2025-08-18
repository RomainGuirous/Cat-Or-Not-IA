import os
import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.callbacks import LambdaCallback
from data.config import IMG_SIZE


def load_data(x_path="data/transformed_data/X.npy", y_path="data/transformed_data/y.npy"):
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Données manquantes. Lance d'abord data/prepare_data.py pour générer {x_path} et {y_path}."
        )
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y


def split_data(X, y):
    # on sépare les données en 2 parties : 80% pour l'entraînement et 20% pour le test (de manière aléatoire)
    # le test permet d'évaluer la performance du modèle sur des images jamais vues
    # la norme est de garder 80% des données pour l'entraînement et 20% pour le test
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model(input_shape):
    # On crée un réseau de neurones convolutif simple (CNN) adapté à la classification d'images

    # input_shape: forme des images d'entrée (hauteur, largeur, canaux)
    # Conv2D: extraction de motifs
    # MaxPooling2D: résumé/compacité
    # Flatten: transforme la matrice en vecteur (liste de nombres)
    # Dense: combine toutes les informations extraites pour apprendre des relations plus globales.
    # Dense(1, activation="sigmoid"): donne une seule valeur entre 0 et 1, probabilité d'être un chat
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(
                1, activation="sigmoid"
            ),  # 1 sortie pour binaire (chat ou pas chat)
        ]
    )
    # On choisit une fonction de perte adaptée à la classification binaire, un optimiseur, et la métrique d'évaluation
    # .compile: configure le modèle pour l'entraînement
    
    # - optimizer: méthode pour ajuster les poids du modèle
    # adam: un optimiseur basé sur les gradients qui ajuste le taux d'apprentissage
    # => la mémoire du modèle est les poids attachés aux neurones
    # => les gradients sont utilisés pour mettre à jour ces poids, empêcher les valeurs extrêmes tout en prenant en compte les erreurs
    
    # - loss: fonction pour évaluer la performance du modèle
    # binary_crossentropy: mesure la différence entre les prédictions et les vraies étiquettes
    
    # - metrics: indicateurs à suivre pendant l'entraînement
    # accuracy: proportion de prédictions correctes
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Callback pour afficher des messages personnalisés lors du .fit
def on_epoch_end(epoch, logs):
    print(f"\n--- Époque {epoch + 1} terminée ---")
    print(f"  - Précision entraînement (accuracy) : {logs['accuracy']:.2f}")
    print(f"  - Perte entraînement (loss) : {logs['loss']:.4f}")
    print(f"  - Précision validation (val_accuracy) : {logs['val_accuracy']:.2f}")
    print(f"  - Perte validation (val_loss) : {logs['val_loss']:.4f}")

def main():
    # 1. Charger les données prétraitées
    X, y = load_data()

    # 2. Séparer en train/test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Définir le modèle
    # On préfère utiliser la forme réelle des données si disponibles (X_train.shape[1:]).
    # Sinon, on construit l'input_shape à partir de IMG_SIZE et on suppose 3 canaux RGB.
    if X_train is not None and X_train.ndim == 4:
        input_shape = X_train.shape[1:]
    else:
        # IMG_SIZE peut être (width, height) ou (height, width) selon l'usage —
        # la convention la plus sûre ici est d'interpréter comme (height, width) si possible.
        try:
            h, w = IMG_SIZE
        except Exception:
            # si IMG_SIZE mal défini, tomber en défaut raisonnable
            h, w = 128, 128
        input_shape = (h, w, 3)

    model = build_model(input_shape=input_shape)

    # 4-5. Entraîner le modèle et valider

    # personnaliser les messages de fin d'époque
    custom_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    
    # epochs: nombre d'itérations sur l'ensemble des données d'entraînement
    # validation_data: données pour évaluer le modèle pendant l'entraînement
    # callbacks: fonctions appelées à la fin de chaque époque
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[custom_callback])

    # 6. Évaluer le modèle
    
    
    # On teste le modèle sur les données de test pour voir sa performance
    # loss: mesure la performance du modèle (relative à la fonction de perte, loss dans model.compile)
    # acc: proportion de prédictions correctes (relatif au paramètre metrics dans model.compile)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.2f}")

    # 7. Sauvegarder le modèle
    model.save("data/transformed_data/cat_or_not_model.keras")

    # Chaque étape est commentée dans le code pour expliquer son utilité.


if __name__ == "__main__":
    main()
