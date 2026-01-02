"""Phase 2 Streamlit demo app.

Runs live inference with the fine‑tuned transformer and shows SHAP
token‑level explanations for the predicted "depressed" probability.
"""

from __future__ import annotations

import os
import sys
from typing import List

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)  # phase2_transformer/
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
	sys.path.insert(0, SRC_DIR)

from model_loader import TransformerPredictor, load_model
from shap_explainer import ShapResources, build_explainer, explain_text


MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "output", "models"))
PHASE_ROOT = BASE_DIR
REPO_ROOT = os.path.abspath(os.path.join(PHASE_ROOT, ".."))
DEFAULT_DATA = os.path.join(
	REPO_ROOT, "data", "raw", "depression_dataset_reddit_cleaned.csv"
)


def _base_css() -> str:
	return """
	<style>
	body {background-color: #f3f4f6;}
	.app-header {font-size: 1.9rem; font-weight: 700; margin: 0 0 0.25rem 0}
	.app-sub    {color: #6b7280; margin-bottom: 1rem}
	.card {background: white; padding: 1.1rem 1.3rem; border-radius: 14px;
		   border: 1px solid #e5e7eb; box-shadow: 0 2px 14px rgba(0,0,0,0.05);
		   margin-bottom: 0.75rem}
	.pill {display:inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
		   font-weight:600; font-size:0.9rem}
	.pill-green {background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0}
	.pill-red   {background:#fef2f2; color:#991b1b; border:1px solid #fecaca}
	.bar {height: 10px; border-radius: 999px; background: #e5e7eb; overflow: hidden}
	.bar > span {display:block; height: 100%; background: linear-gradient(90deg,#22c55e,#ef4444);}
	.token {display:inline-block; padding:0.15rem 0.35rem; margin:0 0.1rem 0.15rem 0;
			border-radius:0.35rem; font-size:0.86rem;}
	</style>
	"""


def _header(title: str, subtitle: str) -> str:
	return (
		f"<div class='app-header'>{title}</div>"
		f"<div class='app-sub'>{subtitle}</div>"
	)


def _prediction_card(label: str, prob: float) -> str:
	pill_class = "pill-red" if label.startswith("Depressed") else "pill-green"
	return (
		f"<div class='card'><div class='pill {pill_class}'>{label}</div>"
		f"<div style='margin-top:0.75rem'>Predicted probability of depression</div>"
		f"<div class='bar'><span style='width:{prob*100:.0f}%'></span></div>"
		f"<div style='margin-top:0.5rem; color:#6b7280'>{prob*100:.1f}%</div>"
		f"</div>"
	)


def _color_for_value(v: float) -> str:
	# Blue for protective (negative), red for risk (positive).
	if v >= 0:
		alpha = min(0.12 + abs(v) * 0.25, 0.9)
		return f"rgba(248, 113, 113, {alpha})"  # red-400
	else:
		alpha = min(0.12 + abs(v) * 0.25, 0.9)
		return f"rgba(96, 165, 250, {alpha})"  # blue-400


def _token_html(tokens_with_values):
	spans = []
	for token, val in tokens_with_values:
		bg = _color_for_value(val)
		title = f"SHAP: {val:.4f}"
		spans.append(
			f"<span class='token' style='background:{bg}' title='{title}'>{token}</span>"
		)
	return "".join(spans)


@st.cache_resource(show_spinner=False)
def _load_predictor() -> TransformerPredictor:
	return load_model(MODEL_DIR)


@st.cache_resource(show_spinner=False)
def _load_background_texts(max_samples: int = 80) -> List[str]:
	if not os.path.exists(DEFAULT_DATA):
		return [
			"I'm feeling okay today and trying to stay positive.",
			"Lately I've been struggling to find motivation for anything.",
			"Things feel heavy but I am talking to friends about it.",
		]
	df = pd.read_csv(DEFAULT_DATA)
	texts = df.get("clean_text")
	if texts is None:
		return []
	sample = texts.astype(str).dropna().sample(
		n=min(max_samples, len(texts)), random_state=42
	)
	return sample.tolist()


@st.cache_resource(show_spinner=False)
def _build_shap(_predictor: TransformerPredictor) -> ShapResources | None:
	bg = _load_background_texts()
	if not bg:
		return None
	return build_explainer(_predictor, bg)


def main() -> None:
	st.set_page_config(
		page_title="MIND-AID Phase 2",
		page_icon="🧠",
		layout="wide",
	)

	st.markdown(_base_css(), unsafe_allow_html=True)
	st.markdown(
		_header(
			"🧠 MIND-AID — Phase 2 Transformer",
			"Transformer-based depression signal detection with SHAP explanations",
		),
		unsafe_allow_html=True,
	)

	try:
		predictor = _load_predictor()
	except FileNotFoundError as e:
		st.error(str(e))
		st.info(
			"Train the Phase 2 model first: `python src/train_transformer.py` "
			"inside the phase2_transformer folder."
		)
		return

	shap_res = _build_shap(predictor)

	if "user_text" not in st.session_state:
		st.session_state.user_text = ""

	with st.sidebar:
		st.header("Examples")
		examples = {
			"Neutral": "Had a long day but a walk helped clear my head.",
			"Mild": "Feeling low lately but still managing to get by.",
			"Flag": "Everything feels pointless and I can barely get out of bed.",
			"Severe": "I can't see any future for myself and nothing seems worth it.",
		}
		choice = st.selectbox("Preset", list(examples.keys()))
		if st.button("Use example", type="secondary"):
			st.session_state.user_text = examples[choice]
			st.toast("Loaded example", icon="✅")
		st.caption("Model directory:")
		st.code(MODEL_DIR, language="text")

	col1, col2 = st.columns([1, 1])

	with col1:
		st.markdown("### Enter text")
		user_text = st.text_area(
			"Input",
			key="user_text",
			height=220,
			label_visibility="collapsed",
		)
		predict_clicked = st.button("🔮 Predict & Explain", type="primary")

	with col2:
		st.markdown("### Prediction & explanation")
		if predict_clicked:
			text = (user_text or "").strip()
			if not text:
				st.warning("Please enter some text first.")
			else:
				with st.spinner("Running transformer and SHAP explanations..."):
					labels, pos_probs = predictor.predict([text])
					label_id = int(labels[0])
					prob = float(pos_probs[0])
					label_str = predictor.label_map[label_id]

					st.markdown(
						_prediction_card(label_str, prob),
						unsafe_allow_html=True,
					)

					if shap_res is None:
						st.info(
							"SHAP background data unavailable; only model prediction is shown."
						)
					else:
						top_tokens, _ = explain_text(shap_res, text)
						st.markdown("#### Token contributions")
						st.markdown(
							_token_html(top_tokens),
							unsafe_allow_html=True,
						)
						st.caption(
							"Red tokens push the prediction towards depression; "
							"blue tokens push away. Intensity reflects contribution size."
						)
		else:
			st.write("Click **Predict & Explain** to see output.")

	st.markdown("---")
	st.caption(
		"Research prototype only — not for clinical use. If you or someone "
		"you know is in crisis, please seek professional help immediately."
	)


if __name__ == "__main__":  # pragma: no cover
	main()
