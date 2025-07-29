import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# 1. Charger les données prétraitées
# (Supposons que X et y sont déjà créés par prepare_data.py)
# Ici, on les chargerait depuis un fichier .npy ou on les importe directement si dans le même script
# X = ...
# y = ...

# 2. Séparer en train/test
# On sépare les données pour évaluer la performance du modèle sur des images jamais vues
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Définir le modèle
# On crée un réseau de neurones convolutif simple (CNN) adapté à la classification d'images
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=X_train.shape[1:]),
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

# 4. Compiler le modèle
# On choisit une fonction de perte adaptée à la classification binaire, un optimiseur, et la métrique d'évaluation
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 5. Entraîner le modèle
# On entraîne le modèle sur les images d'entraînement, on valide sur les images de test
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 6. Évaluer le modèle
# On mesure la performance finale sur le jeu de test
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")

# 7. Sauvegarder le modèle
# Pour pouvoir le réutiliser plus tard sans tout réentraîner
model.save("cat_or_not_model.keras")

# Chaque étape est commentée dans le code pour expliquer son utilité.
