# src/model_utils.py
import os
import math
import joblib

LABEL_MAP = {0: "Not depressed / Normal", 1: "Depressed / Needs attention"}

def load_models(model_dir):
    tfidf_path = os.path.join(model_dir, "tfidf_baseline.joblib")
    clf_path = os.path.join(model_dir, "linear_svc_baseline.joblib")

    if not os.path.exists(tfidf_path) or not os.path.exists(clf_path):
        raise FileNotFoundError(
            "Model files not found. Please train first using `py src/train.py`.\n"
            f"Expected: {tfidf_path} and {clf_path}"
        )

    tfidf_loaded = joblib.load(tfidf_path)
    clf_loaded = joblib.load(clf_path)
    return tfidf_loaded, clf_loaded


def predict_text(text: str, tfidf, clf):
    x = tfidf.transform([text])
    pred = int(clf.predict(x)[0])
    try:
        score = float(clf.decision_function(x)[0])
    except Exception:
        score = 0.0
    prob = 1.0 / (1.0 + math.exp(-score))
    return pred, score, prob
