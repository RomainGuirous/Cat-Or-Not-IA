import os
import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from data.utils import pil_to_array
from data.config import IMG_SIZE

# Paramètres
MODEL_PATH = (
    "data/transformed_data/cat_or_not_model.keras"  # chemin attendu du modèle entraîné
)

st.title("Cat-or-Not — Testeur d'image")
st.write(
    "Uploadez une image, l'application la prétraite et affiche la prédiction si un modèle est disponible."
)


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
    st.warning(
        f"Modèle introuvable à '{MODEL_PATH}'. Veuillez entraîner et sauvegarder le modèle sous ce chemin, ou modifier MODEL_PATH dans le fichier."
    )

uploaded = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg", "bmp"])


def preprocess_image(pil_img, size=None, model=None):
    """Prétraite une image PIL et l'ajuste à la taille attendue par le modèle.

    Si `model` est fourni, on utilise la forme d'entrée du modèle (model.input_shape)
    pour déterminer la taille de redimensionnement (largeur, hauteur). Sinon on utilise
    `size` si fourni, puis `IMG_SIZE`.
    """
    # Déterminer la taille cible au format attendu par PIL (width, height)
    target = None
    if model is not None:
        try:
            in_shape = model.input_shape
            # input_shape peut être une liste (cas multi-input) -> prendre le premier
            if (
                isinstance(in_shape, (list, tuple))
                and len(in_shape) > 0
                and isinstance(in_shape[0], (list, tuple))
            ):
                in_shape = in_shape[0]
            # in_shape attendu: (None, height, width, channels)
            if in_shape is not None and len(in_shape) >= 3:
                h = in_shape[1]
                w = in_shape[2]
                if h is not None and w is not None:
                    target = (int(w), int(h))
        except Exception:
            target = None

    if target is None:
        if size is not None:
            target = size
        else:
            target = IMG_SIZE

    arr = pil_to_array(pil_img, size=target)
    arr = np.expand_dims(arr, axis=0)  # ajoute la dimension du batch
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
            x = preprocess_image(img, model=model)
            preds = model.predict(x)

            # Interpréter la sortie du modèle en tenant compte que ici
            # la classe 0 = Chat et la classe 1 = Pas de chat (non-chat).
            probs = None
            pred_idx = None
            try:
                p = np.array(preds).reshape(-1)
                if p.size == 1:
                    # Sigmoïde -> p[0] est la probabilité de la classe '1' (non-chat)
                    prob_class1 = float(p[0])
                    probs = [1.0 - prob_class1, prob_class1]
                    pred_idx = 0 if probs[0] >= 0.5 else 1
                else:
                    # Softmax -> p contient les probabilités par classe dans l'ordre des indices
                    probs = [float(v) for v in p]
                    pred_idx = int(np.argmax(probs))
            except Exception:
                probs = None
                pred_idx = None

            if probs is None or pred_idx is None:
                st.error("Impossible d'interpréter la sortie du modèle.")
            else:
                CLASS_MAP = {0: "Chat", 1: "Pas de chat"}
                pred_label = CLASS_MAP.get(pred_idx, f"classe_{pred_idx}")
                st.metric(
                    label=f"Prédiction: {pred_label}",
                    value=f"{probs[pred_idx] * 100:.2f}% confiance",
                )
                st.write(
                    "Interprétation: 0 = Chat, 1 = Pas de chat. Seuil utilisé (sigmoïde) = 0.5"
                )

                with st.expander("Détails de la prédiction brut"):
                    st.write(
                        {
                            "raw_output": preds.tolist(),
                            "probs": probs,
                            "pred_idx": pred_idx,
                            "mapping": CLASS_MAP,
                        }
                    )

st.write(
    "\n---\nAstuce: vérifiez que la taille de l'image (IMG_SIZE) et le chemin du modèle (MODEL_PATH) correspondent à ceux utilisés lors de l'entraînement."
)
