"""Phase 2 transformer training script.

Fine‑tunes a Hugging Face transformer for binary classification on the
same dataset schema as Phase 1 (``clean_text``, ``is_depression``), and
saves:

- model/tokenizer under ``phase2_transformer/output/models``
- metrics under ``phase2_transformer/output/metrics``
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	get_linear_schedule_with_warmup,
)

from tokenization import DEFAULT_MAX_LENGTH, DEFAULT_MODEL_NAME


def _set_seed(seed: int = 42) -> None:
	torch.manual_seed(seed)
	np.random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _prepare_datasets(
	csv_path: str,
	tokenizer: AutoTokenizer,
	max_length: int = DEFAULT_MAX_LENGTH,
	test_size: float = 0.15,
	seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
	df = pd.read_csv(csv_path)
	texts = df["clean_text"].astype(str).tolist()
	labels = df["is_depression"].astype(int).values

	X_train, X_val, y_train, y_val = train_test_split(
		texts, labels, test_size=test_size, random_state=seed, stratify=labels
	)

	def encode(texts_list, labels_arr):
		enc = tokenizer(
			texts_list,
			padding=True,
			truncation=True,
			max_length=max_length,
			return_tensors="pt",
		)
		labels_tensor = torch.tensor(labels_arr, dtype=torch.long)
		return TensorDataset(enc["input_ids"], enc["attention_mask"], labels_tensor)

	train_ds = encode(X_train, y_train)
	val_ds = encode(X_val, y_val)
	return train_ds, val_ds


def train(
	data_path: str,
	out_dir: str,
	metrics_dir: Optional[str] = None,
	model_name: str = DEFAULT_MODEL_NAME,
	epochs: int = 2,
	batch_size: int = 16,
	learning_rate: float = 2e-5,
) -> Dict[str, float]:
	"""Fine‑tune a transformer and save artifacts + metrics.

	Returns a dict of key metrics (e.g. accuracy) for convenience.
	"""

	_set_seed(42)

	os.makedirs(out_dir, exist_ok=True)
	if metrics_dir is None:
		metrics_dir = os.path.join(os.path.dirname(out_dir), "metrics")
	os.makedirs(metrics_dir, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_name, num_labels=2
	).to(device)

	train_ds, val_ds = _prepare_datasets(data_path, tokenizer)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size)

	optimizer = AdamW(model.parameters(), lr=learning_rate)
	total_steps = len(train_loader) * epochs
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=int(0.1 * total_steps),
		num_training_steps=total_steps,
	)

	best_val_acc = 0.0
	history = []

	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0

		for batch in train_loader:
			input_ids, attention_mask, labels = [b.to(device) for b in batch]

			optimizer.zero_grad()
			outputs = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				labels=labels,
			)
			loss = outputs.loss
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			scheduler.step()

			total_loss += loss.item()

		avg_train_loss = total_loss / max(len(train_loader), 1)

		# Validation
		model.eval()
		all_labels = []
		all_preds = []

		with torch.no_grad():
			for batch in val_loader:
				input_ids, attention_mask, labels = [b.to(device) for b in batch]
				outputs = model(
					input_ids=input_ids,
					attention_mask=attention_mask,
				)
				logits = outputs.logits
				preds = torch.argmax(logits, dim=-1)
				all_labels.extend(labels.cpu().numpy().tolist())
				all_preds.extend(preds.cpu().numpy().tolist())

		val_acc = accuracy_score(all_labels, all_preds)
		report_str = classification_report(all_labels, all_preds)

		history.append({
			"epoch": epoch,
			"train_loss": avg_train_loss,
			"val_accuracy": float(val_acc),
		})

		# Save best model weights so far
		if val_acc >= best_val_acc:
			best_val_acc = val_acc
			model.save_pretrained(out_dir)
			tokenizer.save_pretrained(out_dir)

		# Also log per-epoch metrics to console
		print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_acc={val_acc:.4f}")
		print(report_str)

	# Final metrics written from the best epoch
	metrics = {
		"best_val_accuracy": float(best_val_acc),
		"epochs": epochs,
		"history": history,
	}

	metrics_path = os.path.join(metrics_dir, "metrics.json")
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	report_path = os.path.join(metrics_dir, "classification_report.txt")
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(f"Best validation accuracy: {best_val_acc:.4f}\n\n")
		f.write("Classification report is from the last epoch above.\n")

	return metrics


def main() -> None:
	"""CLI entrypoint with sensible default paths.

	Defaults mirror Phase 1:

	- Data:   MIND-AID/data/raw/depression_dataset_reddit_cleaned.csv
	- Models: phase2_transformer/output/models
	- Metrics: phase2_transformer/output/metrics
	"""

	parser = argparse.ArgumentParser(
		description="Fine‑tune a transformer baseline for Phase 2.",
	)

	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	phase2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

	default_data = os.path.join(
		repo_root, "data", "raw", "depression_dataset_reddit_cleaned.csv"
	)
	default_models_out = os.path.join(phase2_root, "output", "models")
	default_metrics_out = os.path.join(phase2_root, "output", "metrics")

	parser.add_argument("--data", default=default_data, help="Path to training CSV")
	parser.add_argument(
		"--out",
		dest="out_models",
		default=default_models_out,
		help="Directory to store model/tokenizer artifacts",
	)
	parser.add_argument(
		"--metrics",
		dest="out_metrics",
		default=default_metrics_out,
		help="Directory to store evaluation metrics",
	)
	parser.add_argument(
		"--model-name",
		default=DEFAULT_MODEL_NAME,
		help="Hugging Face model name (e.g. distilbert-base-uncased)",
	)
	parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
	parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
	parser.add_argument(
		"--lr", type=float, default=2e-5, help="Learning rate for AdamW"
	)

	args = parser.parse_args()
	train(
		data_path=args.data,
		out_dir=args.out_models,
		metrics_dir=args.out_metrics,
		model_name=args.model_name,
		epochs=args.epochs,
		batch_size=args.batch_size,
		learning_rate=args.lr,
	)


if __name__ == "__main__":  # pragma: no cover - simple CLI wiring
	main()
