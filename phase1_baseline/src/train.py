"""Phase 1 baseline training script (isolated).

By default, saves:
- model artifacts under `phase1_baseline/output/models/`
- metric outputs under `phase1_baseline/output/metrics/`
"""

import os
import argparse
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score


def train(data_path: str, out_dir: str, metrics_dir: str | None = None):
    """Train TF-IDF + LinearSVC baseline.

    Parameters
    ----------
    data_path:
        Path to input CSV containing columns `clean_text` and `is_depression`.
    out_dir:
        Directory where model artifacts (vectorizer + classifier) are stored.
    metrics_dir:
        Optional directory where evaluation metrics are written. If omitted,
        a sibling `metrics/` directory next to ``out_dir`` is used.
    """

    os.makedirs(out_dir, exist_ok=True)
    if metrics_dir is None:
        metrics_dir = os.path.join(os.path.dirname(out_dir), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    X = df["clean_text"].astype(str).values
    y = df["is_depression"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    clf = LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)
    clf.fit(X_train_tfidf, y_train)
    joblib.dump(tfidf, os.path.join(out_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(clf, os.path.join(out_dir, "svm_model.pkl"))
    y_pred = clf.predict(X_test_tfidf)
    report_str = classification_report(y_test, y_pred)
    print(report_str)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": report_str,
    }

    # Human-readable report
    with open(os.path.join(metrics_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_str)

    # Structured metrics for downstream use
    with open(os.path.join(metrics_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train Phase 1 TF-IDF + LinearSVC baseline.")
    # Repo root (MIND-AID/) for locating shared data
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Phase 1 baseline directory (this phase only)
    phase1_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    default_data = os.path.join(
        repo_root, "data", "raw", "depression_dataset_reddit_cleaned.csv"
    )
    default_models_out = os.path.join(phase1_root, "output", "models")
    default_metrics_out = os.path.join(phase1_root, "output", "metrics")

    parser.add_argument("--data", default=default_data, help="Path to CSV")
    parser.add_argument(
        "--out",
        dest="out_models",
        default=default_models_out,
        help="Output directory for model artifacts (vectorizer + classifier)",
    )
    parser.add_argument(
        "--metrics",
        dest="out_metrics",
        default=default_metrics_out,
        help="Output directory for evaluation metrics",
    )
    args = parser.parse_args()
    train(args.data, args.out_models, args.out_metrics)


if __name__ == "__main__":
    main()
