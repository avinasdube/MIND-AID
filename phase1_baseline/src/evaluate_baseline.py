"""Evaluate Phase 1 TF‑IDF + LinearSVC baseline on the full dataset.

This script loads the trained Phase 1 model from `output/models`, runs it
on the shared CSV under `data/raw/`, and writes metrics/plots under
`output/metrics`.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Ensure repo-root packages (like `common`) and the phase1_baseline `src`
# package are importable when running this script directly.
CURRENT_DIR = os.path.dirname(__file__)
PHASE1_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

for path in (PHASE1_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.metrics import summarize_classification
from src.model_utils import LABEL_MAP, load_models


def evaluate(csv_path: str, model_dir: str, metrics_dir: str | None = None) -> Dict:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    phase1_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if metrics_dir is None:
        metrics_dir = os.path.join(phase1_root, "output", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    tfidf, clf = load_models(model_dir)

    df = pd.read_csv(csv_path)
    texts = df["clean_text"].astype(str).tolist()
    labels = df["is_depression"].astype(int).values

    X = tfidf.transform(texts)
    preds = clf.predict(X)

    # Try to obtain a continuous score for ROC/PR curves
    try:
        scores = clf.decision_function(X)
    except Exception:
        try:
            proba = clf.predict_proba(X)[:, 1]
            scores = proba
        except Exception:
            scores = preds.astype(float)

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)

    metrics = {
        "accuracy": float(acc),
        "n_samples": int(len(labels)),
        "confusion_matrix": cm.tolist(),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }

    # Save metrics + text report
    with open(os.path.join(metrics_dir, "eval_metrics_phase1.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(metrics_dir, "eval_report_phase1.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Common style
    plt.style.use("seaborn-v0_8")

    # Modern confusion matrix: row-normalised heatmap + counts/percentages
    class_names = ["Not depressed / Normal", "Depressed / Needs attention"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Slightly wider than tall for better label fit
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    cmap = plt.cm.get_cmap("YlGnBu")  # colorful but still perceptually ordered
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_title(f"Phase 1 Confusion Matrix (Accuracy = {acc:.2%})", fontsize=15, pad=16)
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    # Equal cell aspect for a clean grid
    ax.set_aspect("equal")

    # Gridlines between cells for a cleaner look
    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each cell with count and row percentage
    row_sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            pct = cm_norm[i, j] * 100.0 if row_sums[i] > 0 else 0.0
            text_color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{value}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="semibold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.ax.set_ylabel("Row-normalised proportion", rotation=90, fontsize=11)

    fig.tight_layout()
    cm_path = os.path.join(metrics_dir, "confusion_matrix_phase1.png")
    fig.savefig(cm_path, dpi=260)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}", color="#2563eb", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate", fontsize=12)
    ax.set_title("Phase 1 ROC curve", fontsize=14)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(metrics_dir, "roc_phase1.png"), dpi=220)
    plt.close(fig)

    # Precision–recall curve
    prec, rec, _ = precision_recall_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, color="#16a34a", linewidth=2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Phase 1 Precision–Recall curve", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(metrics_dir, "pr_curve_phase1.png"), dpi=220)
    plt.close(fig)

    return metrics


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    phase1_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    default_data = os.path.join(
        repo_root, "data", "raw", "depression_dataset_reddit_cleaned.csv"
    )
    default_model_dir = os.path.join(phase1_root, "output", "models")
    default_metrics_dir = os.path.join(phase1_root, "output", "metrics")

    metrics = evaluate(default_data, default_model_dir, default_metrics_dir)
    print("Phase 1 eval metrics:", metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
