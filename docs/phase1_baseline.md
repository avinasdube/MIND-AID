# Phase 1 Baseline: TF‑IDF + LinearSVC

This document provides a complete, high‑detail reference for the Phase 1 baseline of the MIND‑AID project. It covers objectives, architecture, data, environment setup, training and inference workflows, the demo app, model artifacts, testing, CI, troubleshooting, and FAQs. It is intended for developers and analysts who want to run, understand, and extend the baseline.

## Objectives
- Deliver a simple, reliable baseline for text classification on mental‑health discussion data.
- Use interpretable classical ML components: TF‑IDF vectorizer and LinearSVC classifier.
- Establish reproducible data paths, training outputs, and a small demo UI.
- Provide clean per‑phase isolation with its own virtual environment.

## Repository Layout (Relevant Paths)
- `data/raw/depression_dataset_reddit_cleaned.csv`: Input CSV data (text + labels).
- `models/phase1/`: Output directory for trained artifacts.
- `phase1_baseline/src/`: Training and utility code.
- `phase1_baseline/app/`: Streamlit demo app.
- `docs/phase1_baseline.md`: This documentation.

## Architecture Overview
- **Tokenizer/Featurizer**: `scikit‑learn` `TfidfVectorizer` transforms raw text to sparse TF‑IDF features.
- **Classifier**: `LinearSVC` (`scikit‑learn`) learns a linear hyperplane for classification.
- **Persistence**: Artifacts saved via `joblib` for deterministic reload in inference/app.
- **Utilities**: `src/model_utils.py` provides model loading and `predict_text()` helpers.
- **Demo App**: `Streamlit` UI (`phase1_baseline/app/app.py`) wraps prediction for quick testing.

## Data
- Expected file: `data/raw/depression_dataset_reddit_cleaned.csv`
- Required columns:
  - `text`: String input for classification.
  - `label`: Integer or categorical class label.
- Basic assumptions:
  - Minimal preprocessing; TF‑IDF handles tokenization, stopwords optional (default scikit‑learn behavior).
  - Train/validation split handled in script or via simple `sklearn` utilities.

## Environment Setup (Windows Git Bash)
Use a per‑phase virtual environment to avoid conflicts.

```bash
cd "d:/MindAid - Project/MIND-AID/phase1_baseline"
python -m venv .venv
source .venv/Scripts/activate
python -V
pip install --upgrade pip
pip install -r ../requirements.txt
```

Notes:
- Interpreter path should be `phase1_baseline/.venv` in VS Code (`Python: Select Interpreter`).
- Avoid a root `.venv` for this project; use per‑phase only.

## Training Workflow
The training script builds the TF‑IDF vectorizer and trains a LinearSVC, then saves artifacts.

- Script: `phase1_baseline/src/train.py`
- Default paths:
  - Input CSV: `../data/raw/depression_dataset_reddit_cleaned.csv`
  - Output dir: `../models/phase1`
- Saved artifacts:
  - `tfidf_vectorizer.pkl`
  - `svm_model.pkl`

Run training:
```bash
cd "d:/MindAid - Project/MIND-AID/phase1_baseline"
source .venv/Scripts/activate
python src/train.py \
  --data "../data/raw/depression_dataset_reddit_cleaned.csv" \
  --out "../models/phase1"
```

Expected console output includes a `classification_report` and summary metrics.

## Inference Utilities
- File: `phase1_baseline/src/model_utils.py`
- Key functions:
  - `load_models(model_dir)`: Loads TF‑IDF and SVM artifacts; raises clear errors if missing.
  - `predict_text(text, vectorizer, model)`: Returns `(pred_label, score, prob)`; `prob` is a monotonic transform of the decision function (not calibrated probability).
- Usage example:
```python
from model_utils import load_models, predict_text
vec, clf = load_models("../models/phase1")
pred, score, prob = predict_text("sample input", vec, clf)
```

## Demo App (Streamlit)
- Entry: `phase1_baseline/app/app.py`
- Behavior:
  - Adds `phase1_baseline/src` to `sys.path` to import `model_utils` and `ui`.
  - Loads models from `../models/phase1`.
  - Simple text area for input and a result card.

Run the app:
```bash
cd "d:/MindAid - Project/MIND-AID/phase1_baseline"
source .venv/Scripts/activate
streamlit run app/app.py
```

## Models and Artifacts
- Location: `models/phase1/`
- Contents after training:
  - `tfidf_vectorizer.pkl`: The trained `TfidfVectorizer`.
  - `svm_model.pkl`: The trained `LinearSVC` classifier.
- Versioning:
  - For experiment tracking, append timestamps or use subfolders per run.
  - Consider adding metrics JSON (`metrics.json`) for quick comparisons.

## Configuration Options
`src/train.py` commonly supports:
- `--data`: Path to CSV input.
- `--out`: Directory to save artifacts (created if missing).
- Optional toggles (if implemented): `--ngram-range`, `--min-df`, `--max-features`, `--class-weight`.

## Testing
- Minimal smoke tests verify training writes artifacts.
- Example:
  - `phase1_baseline/tests/test_train.py`: Ensures `tfidf_vectorizer.pkl` and `svm_model.pkl` exist after a run.
- Run tests:
```bash
cd "d:/MindAid - Project/MIND-AID/phase1_baseline"
source .venv/Scripts/activate
pytest -q
```

## CI
- GitHub Actions matrix runs per phase with isolated venvs.
- Phase 1 job:
  - Set up venv, install `requirements.txt`, run `pytest`.
- Artifacts are not uploaded by default; enable if needed for deployment.

## Extend/Customize
- Replace `LinearSVC` with `LogisticRegression` for probabilistic outputs (requires `CalibratedClassifierCV` for calibrated probabilities).
- Tune `TfidfVectorizer` parameters (`stop_words`, `ngram_range`, `min_df`, `max_df`).
- Add data cleaning (lowercasing, punctuation removal, domain‑specific token filters).
- Integrate cross‑validation and hyperparameter search (`GridSearchCV` or `RandomizedSearchCV`).

## Troubleshooting
- "ModuleNotFoundError: No module named 'src'" when launching the app:
  - Ensure you run `streamlit` from `phase1_baseline` and the app adds `src` to `sys.path`.
- "FileNotFoundError" for input CSV:
  - Verify `data/raw/depression_dataset_reddit_cleaned.csv` exists and the `--data` path is correct relative to `phase1_baseline/src/train.py`.
- Virtual environment mismatches:
  - Check `python -c "import sys; print(sys.executable)"` points inside `phase1_baseline/.venv`.
  - In VS Code, select the interpreter under `phase1_baseline/.venv`.
- Streamlit fails to start:
  - Reinstall deps: `pip install -r ../requirements.txt` within the active venv.
  - Upgrade `pip` and confirm `streamlit` is installed.

## Frequently Asked Questions
- "Why TF‑IDF + LinearSVC?"
  - Fast, strong baseline, interpretable feature space, low resource usage.
- "Are probabilities calibrated?"
  - No; `prob` is a monotonic transform of decision function. Use calibration for true probabilities.
- "Can I swap in transformers later?"
  - Yes; Phase 2 is reserved for that. Keep Phase 1 intact for baseline comparisons.
- "Where do I change vectorizer settings?"
  - In `src/train.py` and/or a config module; adjust `TfidfVectorizer` args.

## Quick Commands
Common commands for Phase 1 baseline (Windows Git Bash):

```bash
# Setup
cd "d:/MindAid - Project/MIND-AID/phase1_baseline" && python -m venv .venv && source .venv/Scripts/activate && pip install --upgrade pip && pip install -r ../requirements.txt

# Train
python src/train.py --data "../data/raw/depression_dataset_reddit_cleaned.csv" --out "../models/phase1"

# Run app
streamlit run app/app.py

# Test
pytest -q
```
