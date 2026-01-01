import os
import sys
import streamlit as st
# Ensure `src` is importable when running from app directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(APP_DIR), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model_utils import load_models, predict_text, LABEL_MAP
import ui as ui

st.set_page_config(page_title="MIND-AID Phase 1", page_icon="🧠", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "output", "models"))

try:
    tfidf, clf = load_models(MODEL_DIR)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.markdown(ui.base_css(), unsafe_allow_html=True)
st.markdown(ui.header("🧠 MIND-AID — Phase 1 Baseline", "TF‑IDF + LinearSVC"), unsafe_allow_html=True)

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

with st.sidebar:
    st.header("Examples")
    examples = {
        "Neutral": "Had a long day but a walk helped.",
        "Mild": "Feeling a bit low lately but trying to stay active.",
        "Flag": "Everything feels pointless and I have no energy.",
        "Severe": "I can't see any reason to go on like this."}
    choice = st.selectbox("Preset", list(examples.keys()))
    if st.button("Use example", type="secondary"):
        st.session_state.user_text = examples[choice]
        st.toast("Loaded example", icon="✅")
    st.caption("Models directory:")
    st.code(MODEL_DIR, language="text")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Enter text")
    user_text = st.text_area("Input", key="user_text", height=220, label_visibility="collapsed")
    predict_clicked = st.button("🔮 Predict", type="primary")

with col2:
    st.markdown("### Prediction")
    if predict_clicked:
        text = (user_text or "").strip()
        if not text:
            st.warning("Please enter some text first.")
        else:
            pred, score, prob = predict_text(text, tfidf, clf)
            label = LABEL_MAP.get(pred, str(pred))
            st.markdown(ui.prediction_card_html(label, pred, prob, score), unsafe_allow_html=True)
            st.info("Baseline demo only. Not for clinical use.")
    else:
        st.write("Click Predict to see output.")

st.markdown("---")
st.caption("Train first with `python src/train.py` inside phase1_baseline.")
