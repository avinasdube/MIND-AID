# MIND-AID — Current Progress (Simplified Technical Explainer)

This document explains what’s built so far, how it works under the hood, and how to run it. The goal is to keep it simple but technically correct for a reviewer.

## 1) What It Does (Now)

- Detects whether a short text (e.g., Reddit-style post) indicates depression.
- Uses a classic, strong baseline for text classification: TF‑IDF features + Linear Support Vector Classifier (LinearSVC).
- Provides a small web UI (Streamlit) to try inputs and view predictions interactively.

## 2) Data Used

- File: `data/raw/depression_dataset_reddit_cleaned.csv`
- Columns used by the current training script:
  - `clean_text`: pre-cleaned input text.
  - `is_depression`: label (0 = not depressed/normal, 1 = depressed/flag).
- Train/validation split: 85% train / 15% test, stratified by label (keeps class balance consistent).

## 3) Features (How Text Becomes Numbers)

- TF‑IDF Vectorizer from scikit-learn (`TfidfVectorizer`).
- Settings: up to 20,000 features, unigrams and bigrams (`ngram_range=(1,2)`), English stop words removed.
- Intuition: words (and word pairs) that are informative for the label get higher weights.

## 4) Model Choice (Why LinearSVC?)

- Model: `LinearSVC` with `class_weight='balanced'`.
- Why this baseline:
  - Works very well with high-dimensional sparse TF‑IDF features.
  - Fast to train and predict; easy to deploy.
  - `class_weight='balanced'` compensates if one class is rarer.
- Output: LinearSVC gives a decision score (distance from hyperplane). In the app, we map that score through a sigmoid to show a human-friendly “confidence” bar. Note: that confidence is not calibrated probability (good enough for a demo, but not for clinical use).

## 5) Training Pipeline (Code)

- Script: `src/train.py`
  1. Load CSV, take `clean_text` and `is_depression`.
  2. Split into train/test (85/15, stratified).
  3. Fit TF‑IDF on train; transform train/test.
  4. Train `LinearSVC` on TF‑IDF train features.
  5. Save artifacts (with `joblib`) into `models/`:
     - `tfidf_baseline.joblib`
     - `linear_svc_baseline.joblib`
  6. Print `classification_report`; show confusion matrix heatmap (for quick sanity-check).

## 6) Saved Artifacts (What’s Deployed)

- `models/tfidf_baseline.joblib` — the fitted TF‑IDF vectorizer.
- `models/linear_svc_baseline.joblib` — the trained LinearSVC classifier.

## 7) Inference Flow (Prediction)

- Function: `src/model_utils.py::predict_text`
  1. Transform the raw input text using the loaded TF‑IDF.
  2. Get predicted class from the LinearSVC.
  3. Extract decision score and pass through a sigmoid to show a confidence-like percentage.
- Labels:
  - `0` → "Not depressed / Normal"
  - `1` → "Depressed / Needs attention"

## 8) App / Deployment (How You Use It Now)

- App: `src/app.py` (Streamlit)
  - Loads the saved TF‑IDF + LinearSVC from `models/`.
  - Text area to paste input.
  - Predict button → shows label pill, confidence bar, and decision score.
  - Minimal styling via `src/ui.py`.
- This is a demo UI to explore behavior; not a medical device.

## 9) How to Run (Windows bash.exe)

Prerequisites: Python 3.10+ recommended.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source ".venv/Scripts/activate"

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train (writes joblib files into models/)
py src/train.py

# 4) Launch the demo app
python -m streamlit run src/app.py
```

If you use PowerShell/CMD instead of bash, activate with `.venv\Scripts\activate` (no `source`).

## 10) What’s Done

- Baseline text classifier (TF‑IDF + LinearSVC) implemented and trained.
- Artifacts saved and loaded for inference.
- Minimal interactive app deployed with Streamlit.
- Basic evaluation printed (`classification_report`) and confusion matrix shown during training.

## 11) What’s Next (Good Talking Points)

- Probabilities: add calibration (e.g., `CalibratedClassifierCV`) for better probability estimates.
- Evaluation: add cross-validation, ROC‑AUC, precision‑recall curves; save metrics/plots.
- Data pipeline: move text cleaning and splits into a reusable `data_processing.py` with tests.
- Error analysis: inspect false positives/negatives to refine features.
- Model upgrades: try logistic regression/SGD baselines and then transformer-based encoders.
- MLOps: version datasets/artifacts, log experiments (e.g., MLflow), package the app.
- Safety/ethics: add disclaimers, bias checks, and human-in-the-loop guidance.

## 12) Simple Architecture View

```text
CSV → TF‑IDF (fit) → LinearSVC (fit)
  |         |              |
  └─ save tfidf.joblib    └─ save model.joblib

User Text → load tfidf + model → TF‑IDF (transform) → LinearSVC (predict) → Label + Confidence (UI)
```

## 13) Key Files

- Training: `src/train.py`
- Inference utils: `src/model_utils.py`
- App UI: `src/app.py`, `src/ui.py`
- Artifacts: `models/`
- Data: `data/raw/depression_dataset_reddit_cleaned.csv`

---
Tip: This is a baseline demo to show end‑to‑end functionality. It’s intentionally simple and fast, and is a solid foundation to iterate from.
