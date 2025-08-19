#!/usr/bin/env bash

# si une commande retourne un code de sortie non‑zéro, le script s'arrête immédiatement
set -e

python -m data.prepare_data
python -m train_keras
streamlit run app.py
