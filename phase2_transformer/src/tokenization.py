"""Tokenization utilities for Phase 2 transformer models.

These helpers wrap ``transformers.AutoTokenizer`` so that both the
training script and the demo app use a consistent configuration
(``model_name``, ``max_length`` etc.).
"""

from __future__ import annotations

from typing import Iterable, List

from transformers import AutoTokenizer


DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 160


def get_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
	"""Return a Hugging Face tokenizer for the requested model.

	Parameters
	----------
	model_name:
		Hugging Face model identifier, e.g. ``"distilbert-base-uncased"``.
	"""

	return AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(
	tokenizer,
	texts: Iterable[str],
	max_length: int = DEFAULT_MAX_LENGTH,
):
	"""Batch-tokenize a collection of texts.

	Returns a dict of tensors compatible with transformer models.
	"""

	return tokenizer(
		list(texts),
		padding=True,
		truncation=True,
		max_length=max_length,
		return_tensors="pt",
	)


__all__: List[str] = [
	"DEFAULT_MODEL_NAME",
	"DEFAULT_MAX_LENGTH",
	"get_tokenizer",
	"tokenize_texts",
]
