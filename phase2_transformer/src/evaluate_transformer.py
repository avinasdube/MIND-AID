"""Standalone evaluation script for Phase 2 transformer models.

This module loads a trained model from ``output/models`` and evaluates
it on a given CSV (same schema as training), writing metrics under
``output/metrics``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

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

from model_loader import TransformerPredictor, get_default_model_dir, load_model


def _batched_predict(
	predictor: TransformerPredictor,
	texts: List[str],
	batch_size: int = 32,
):
	all_preds: List[int] = []
	all_pos_probs: List[float] = []
	for i in range(0, len(texts), batch_size):
		chunk = texts[i : i + batch_size]
		labels, pos_probs = predictor.predict(chunk)
		all_preds.extend([int(x) for x in labels])
		all_pos_probs.extend([float(x) for x in pos_probs])
	return np.array(all_preds), np.array(all_pos_probs)


def evaluate(
	csv_path: str,
	model_dir: str | None = None,
	metrics_dir: str | None = None,
	batch_size: int = 32,
) -> Dict:
	"""Evaluate a trained Phase 2 model on the provided dataset."""

	if model_dir is None:
		model_dir = get_default_model_dir()

	phase2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	if metrics_dir is None:
		metrics_dir = os.path.join(phase2_root, "output", "metrics")
	os.makedirs(metrics_dir, exist_ok=True)

	predictor: TransformerPredictor = load_model(model_dir)

	df = pd.read_csv(csv_path)
	texts = df["clean_text"].astype(str).tolist()
	labels = df["is_depression"].astype(int).values

	preds, pos_probs = _batched_predict(predictor, texts, batch_size=batch_size)

	acc = accuracy_score(labels, preds)
	report = classification_report(labels, preds)
	cm = confusion_matrix(labels, preds)
	roc_auc = roc_auc_score(labels, pos_probs)

	metrics = {
		"accuracy": float(acc),
		"n_samples": int(len(labels)),
		"confusion_matrix": cm.tolist(),
		"roc_auc": float(roc_auc),
	}

	with open(os.path.join(metrics_dir, "eval_metrics_phase2.json"), "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	with open(os.path.join(metrics_dir, "eval_report_phase2.txt"), "w", encoding="utf-8") as f:
		f.write(report)

	# Use a consistent modern style
	plt.style.use("seaborn-v0_8")

	# Modern confusion matrix: row-normalised heatmap + counts/percentages
	class_names = ["Not depressed / Normal", "Depressed / Needs attention"]
	cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

	# Slightly wider than tall for better label fit
	fig, ax = plt.subplots(figsize=(6.2, 4.6))
	cmap = plt.cm.get_cmap("YlGnBu")
	im = ax.imshow(cm_norm, cmap=cmap, vmin=0.0, vmax=1.0)

	ax.set_title(f"Phase 2 Confusion Matrix (Accuracy = {acc:.2%})", fontsize=15, pad=16)
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
	fig.savefig(os.path.join(metrics_dir, "confusion_matrix_phase2.png"), dpi=260)
	plt.close(fig)

	# ROC curve
	fpr, tpr, _ = roc_curve(labels, pos_probs)
	fig, ax = plt.subplots(figsize=(5, 5))
	ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}", color="#7c3aed", linewidth=2)
	ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
	ax.set_xlabel("False positive rate", fontsize=12)
	ax.set_ylabel("True positive rate", fontsize=12)
	ax.set_title("Phase 2 ROC curve", fontsize=14)
	ax.legend(loc="lower right")
	fig.tight_layout()
	fig.savefig(os.path.join(metrics_dir, "roc_phase2.png"), dpi=220)
	plt.close(fig)

	# Precision–recall curve
	prec, rec, _ = precision_recall_curve(labels, pos_probs)
	fig, ax = plt.subplots(figsize=(5, 5))
	ax.plot(rec, prec, color="#f97316", linewidth=2)
	ax.set_xlabel("Recall", fontsize=12)
	ax.set_ylabel("Precision", fontsize=12)
	ax.set_title("Phase 2 Precision–Recall curve", fontsize=14)
	fig.tight_layout()
	fig.savefig(os.path.join(metrics_dir, "pr_curve_phase2.png"), dpi=220)
	plt.close(fig)

	print(f"Eval accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")
	print(report)
	return metrics


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate Phase 2 transformer model.")

	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	default_data = os.path.join(
		repo_root, "data", "raw", "depression_dataset_reddit_cleaned.csv"
	)

	parser.add_argument("--data", default=default_data, help="Path to evaluation CSV")
	parser.add_argument(
		"--model-dir",
		default=get_default_model_dir(),
		help="Directory where the trained model/tokenizer are stored",
	)
	parser.add_argument(
		"--metrics-dir",
		default=None,
		help="Directory to write evaluation metrics (defaults to output/metrics)",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=32,
		help="Batch size for evaluation to control memory usage",
	)

	args = parser.parse_args()
	evaluate(args.data, args.model_dir, args.metrics_dir, batch_size=args.batch_size)


if __name__ == "__main__":  # pragma: no cover - CLI wiring
	main()
