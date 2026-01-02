"""Model loading and inference utilities for Phase 2 transformers.

The training script saves a Hugging Face tokenizer and
``AutoModelForSequenceClassification`` under

	``phase2_transformer/output/models``

This module provides a small wrapper around those artifacts so both the
demo app and SHAP explainer can make predictions in a consistent way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tokenization import DEFAULT_MAX_LENGTH


def get_default_model_dir() -> str:
	"""Return the default directory for Phase 2 models.

	By convention, all Phase 2 transformer artifacts live under::

		phase2_transformer/output/models
	"""

	phase2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	return os.path.join(phase2_root, "output", "models")


@dataclass
class TransformerPredictor:
	"""Lightweight prediction wrapper around a fine-tuned transformer.

	Exposes convenience methods for class probabilities and predictions
	over raw text inputs.
	"""

	tokenizer: AutoTokenizer
	model: AutoModelForSequenceClassification
	device: torch.device
	max_length: int = DEFAULT_MAX_LENGTH
	label_map: Tuple[str, str] = ("Not depressed / Normal", "Depressed / Needs attention")

	def _encode(self, texts: Iterable[str]) -> dict:
		enc = self.tokenizer(
			list(texts),
			padding=True,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
		)
		return {k: v.to(self.device) for k, v in enc.items()}

	def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
		"""Return class probabilities with shape ``(n_samples, 2)``."""

		self.model.eval()
		batch = self._encode(texts)
		with torch.no_grad():
			outputs = self.model(**batch)
			logits = outputs.logits
			probs = F.softmax(logits, dim=-1).cpu().numpy()
		return probs

	def predict(self, texts: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
		"""Return predicted labels and probabilities for the positive class.

		Returns
		-------
		labels : np.ndarray
			Integer labels (0 or 1).
		positive_probs : np.ndarray
			Probability assigned to the "depressed" class (index 1).
		"""

		probs = self.predict_proba(texts)
		labels = probs.argmax(axis=1)
		pos_probs = probs[:, 1]
		return labels, pos_probs


def load_model(model_dir: str | None = None) -> TransformerPredictor:
	"""Load the fine-tuned transformer and tokenizer.

	Parameters
	----------
	model_dir:
		Directory containing the saved model and tokenizer. If omitted,
		defaults to :func:`get_default_model_dir`.
	"""

	if model_dir is None:
		model_dir = get_default_model_dir()

	if not os.path.isdir(model_dir):
		raise FileNotFoundError(
			f"Phase 2 model directory not found at '{model_dir}'. "
			"Train the transformer first with `python src/train_transformer.py`."
		)

	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = AutoModelForSequenceClassification.from_pretrained(model_dir)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	return TransformerPredictor(tokenizer=tokenizer, model=model, device=device)


__all__: List[str] = [
	"TransformerPredictor",
	"get_default_model_dir",
	"load_model",
]
