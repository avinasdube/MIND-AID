"""Phase 1 baseline training script (isolated).

Saves artifacts under `models/phase1/`.
"""

import os
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


def train(data_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
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
    print(classification_report(y_test, y_pred))


def main():
    parser = argparse.ArgumentParser(description="Train Phase 1 TF-IDF + LinearSVC baseline.")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_data = os.path.join(repo_root, "data", "raw", "depression_dataset_reddit_cleaned.csv")
    default_out = os.path.join(repo_root, "models", "phase1")
    parser.add_argument("--data", default=default_data, help="Path to CSV")
    parser.add_argument("--out", default=default_out, help="Output directory for artifacts")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
