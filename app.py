import os
import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from data.utils import pil_to_array

# Paramètres
IMG_SIZE = (64, 64)  # Doit correspondre à la taille utilisée pour l'entraînement
MODEL_PATH = "data/transformed_data/cat_model.keras"  # chemin attendu du modèle entraîné

st.title("Cat-or-Not — Testeur d'image")
st.write("Uploadez une image, l'application la prétraite et affiche la prédiction si un modèle est disponible.")

@st.cache_resource
def load_model(path):
    """
    Charge le modèle Keras depuis le chemin spécifié.
    Si le modèle n'existe pas ou ne peut pas être chargé, retourne None.
    
    Args:
        path (str): Chemin vers le fichier du modèle Keras.
        
    Returns:
        tf.keras.Model ou None: Le modèle chargé ou None en cas d'erreur.
    """
    if not os.path.exists(path):
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Impossible de charger le modèle: {e}")
        return None

model = load_model(MODEL_PATH)
if model is None:
    st.warning(f"Modèle introuvable à '{MODEL_PATH}'. Veuillez entraîner et sauvegarder le modèle sous ce chemin, ou modifier MODEL_PATH dans le fichier.")

uploaded = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg", "bmp"]) 

def preprocess_image(pil_img, size=IMG_SIZE):
    # Utilise la fonction partagée pour garantir le même traitement que prepare_data.py
    arr = pil_to_array(pil_img, size=size)
    arr = np.expand_dims(arr, axis=0)  # forme (1, H, W, C)
    return arr

if uploaded is not None:
    try:
        img = Image.open(io.BytesIO(uploaded.read()))
    except Exception:
        st.error("Impossible d'ouvrir l'image téléchargée.")
        img = None

    if img is not None:
        st.image(img, caption="Image chargée", use_column_width=True)

        if model is None:
            st.info("Aucun modèle chargé — uniquement affichage d'image.")
        else:
            x = preprocess_image(img)
            preds = model.predict(x)

            # Interpréter la sortie du modèle (gérer sigmoid et softmax)
            prob_cat = None
            try:
                p = np.array(preds).reshape(-1)
                if p.size == 1:
                    # sigmoïde -> probabilité de la classe positive (1 = chat)
                    prob_cat = float(p[0])
                elif p.size == 2:
                    # softmax -> deuxième entrée = probabilité de la classe '1'
                    prob_cat = float(p[1])
                else:
                    # cas inhabituel: prendre la probabilité de l'indice 1 si présent, sinon max
                    prob_cat = float(p[1]) if p.size > 1 else float(np.max(p))
            except Exception:
                prob_cat = None

            if prob_cat is None:
                st.error("Impossible d'interpréter la sortie du modèle.")
            else:
                label = "Chat" if prob_cat >= 0.5 else "Pas de chat"
                st.metric(label=f"Prédiction: {label}", value=f"{prob_cat*100:.2f}% confiance")
                st.write("Interprétation: 1 = chat, 0 = pas de chat. Seuil choisi = 0.5")

                # Afficher plus d'infos brutes
                with st.expander("Détails de la prédiction brut"):
                    st.write({"raw_output": preds.tolist()})

st.write("\n---\nAstuce: vérifiez que la taille de l'image (IMG_SIZE) et le chemin du modèle (MODEL_PATH) correspondent à ceux utilisés lors de l'entraînement.")
