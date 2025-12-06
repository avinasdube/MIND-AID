import os
import math
import joblib

LABEL_MAP = {0: "Not depressed / Normal", 1: "Depressed / Needs attention"}


def load_models(model_dir):
    """Load TF-IDF vectorizer and SVM model from Phase 1 artifact folder."""
    tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    clf_path = os.path.join(model_dir, "svm_model.pkl")

    missing = []
    if not os.path.exists(tfidf_path):
        missing.append(tfidf_path)
    if not os.path.exists(clf_path):
        missing.append(clf_path)
    if missing:
        raise FileNotFoundError(
            "Model files not found. Train Phase 1 baseline first using `python src/train.py`.\n"
            + "Missing: " + ", ".join(missing)
        )

    tfidf_loaded = joblib.load(tfidf_path)
    clf_loaded = joblib.load(clf_path)
    return tfidf_loaded, clf_loaded


def predict_text(text: str, tfidf, clf):
    """Predict label, decision score and approximate probability for a single input text."""
    x = tfidf.transform([text])
    pred = int(clf.predict(x)[0])
    try:
        score = float(clf.decision_function(x)[0])
    except Exception:
        score = 0.0
    prob = 1.0 / (1.0 + math.exp(-score))
    return pred, score, prob
