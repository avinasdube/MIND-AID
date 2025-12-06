#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py
