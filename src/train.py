"""Training script for baseline LinearSVC model.

This version assumes the input CSV has already been preprocessed and contains
`clean_text` and `is_depression` columns, as in the original setup.
"""

import os
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


def main(data_path=None, models_dir=None):
    # Resolve defaults relative to repository root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not data_path:
        data_path = os.path.join(base_dir, "data", "raw", "depression_dataset_reddit_cleaned.csv")
    if not models_dir:
        models_dir = os.path.join(base_dir, "models")

    # Make sure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df['clean_text'].astype(str).values
    y = df['is_depression'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    clf = LinearSVC(class_weight='balanced', random_state=42, max_iter=10000)
    clf.fit(X_train_tfidf, y_train)

    # Save
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_baseline.joblib"))
    joblib.dump(clf, os.path.join(models_dir, "linear_svc_baseline.joblib"))

    # Eval
    y_pred = clf.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix (Baseline)")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple LinearSVC model with TF-IDF features.")
    parser.add_argument("--data-path", dest="data_path", help="Path to input CSV (optional)")
    parser.add_argument("--models-dir", dest="models_dir", help="Directory to save models (optional)")
    args = parser.parse_args()

    main(data_path=args.data_path, models_dir=args.models_dir)
