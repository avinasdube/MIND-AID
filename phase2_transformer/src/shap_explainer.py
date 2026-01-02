"""SHAP explanation utilities for Phase 2 transformer models.

We use :mod:`shap` with a text masker to compute token-level
contributions for the "depressed" class probability. The goal is to
provide intuitive, per‑token attributions suitable for visualization in
the Streamlit demo app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import shap

from model_loader import TransformerPredictor


@dataclass
class ShapResources:
	"""Container for SHAP explainer + background texts."""

	explainer: shap.Explainer
	background_texts: List[str]


def build_explainer(
	predictor: TransformerPredictor,
	background_texts: Iterable[str],
	max_background: int = 50,
) -> ShapResources:
	"""Create a SHAP text explainer for the given predictor.

	``background_texts`` should be representative inputs from the
	training distribution (e.g., a small sample from the training CSV).
	"""

	bg_list = list(background_texts)[:max_background]
	if not bg_list:
		raise ValueError("background_texts must contain at least one example")

	# We explain the probability of the positive (depression) class.
	def f(texts: List[str]) -> np.ndarray:
		probs = predictor.predict_proba(texts)
		# shap expects 2D array; use positive class as a single output
		return probs[:, 1:2]

	masker = shap.maskers.Text(predictor.tokenizer)
	explainer = shap.Explainer(f, masker)
	return ShapResources(explainer=explainer, background_texts=bg_list)


def explain_text(
	resources: ShapResources,
	text: str,
	max_tokens: int = 15,
) -> Tuple[List[Tuple[str, float]], shap.Explanation]:
	"""Explain a single text and return top token contributions.

	Returns a sorted list of ``(token, contribution)`` pairs (by
	absolute contribution, descending) and the full SHAP Explanation
	object for further visualization if needed.
	"""

	explanation = resources.explainer([text])[0]
	tokens = list(explanation.data)
	values = np.array(explanation.values)

	# Aggregate by token; SHAP for text may operate at the token level
	# already, so this is typically one value per token.
	token_contribs: List[Tuple[str, float]] = list(zip(tokens, values))
	token_contribs.sort(key=lambda tv: abs(tv[1]), reverse=True)

	return token_contribs[:max_tokens], explanation


__all__: List[str] = [
	"ShapResources",
	"build_explainer",
	"explain_text",
]
