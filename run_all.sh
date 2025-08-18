#!/usr/bin/env bash

# si une commande retourne un code de sortie non‑zéro, le script s'arrête immédiatement
set -e

python data/prepare_data.py
python train_keras.py
streamlit run app.py
