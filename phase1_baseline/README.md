# Phase 1 Baseline

Classical ML baseline for MIND-AID using TF-IDF features + LinearSVC.

## Contents

- `src/` training and inference utilities
- `app/` Streamlit demo for baseline model
- `models/phase1/` (created at repo root) stores `tfidf_vectorizer.pkl` and `svm_model.pkl`
- `requirements.txt` minimal deps for the baseline

## Quickstart

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
python src/train.py  # trains and saves vectorizer + model under ../../models/phase1
streamlit run app/app.py
```

### Commands explained

- `python -m venv .venv`: creates a virtual environment inside `phase1_baseline/`.
- `source .venv/Scripts/activate`: activates the venv in Windows Git Bash.
- `pip install -r requirements.txt`: installs baseline dependencies.
- `python src/train.py`: trains TF‑IDF + LinearSVC and saves under `models/phase1/`.
- `streamlit run app/app.py`: launches the baseline demo UI.

### Optional flags

Use a custom dataset path or output directory:

```bash
python src/train.py \
	--data "../data/raw/depression_dataset_reddit_cleaned.csv" \
	--out "../models/phase1"
```

## Notes

- Input CSV: `data/raw/depression_dataset_reddit_cleaned.csv` with columns `clean_text`, `is_depression`.
- Keep this phase frozen once Phase 2 (transformers) development begins.
