# src/app.py
import os
import streamlit as st
from model_utils import load_models, predict_text, LABEL_MAP
import ui

st.set_page_config(page_title="MIND-AID Demo", page_icon="🧠", layout="wide")

# Resolve model directory relative to repo root in a simple way
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

try:
    tfidf, clf = load_models(MODEL_DIR)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Basic styles and header
st.markdown(ui.base_css(), unsafe_allow_html=True)
st.markdown(ui.header("🧠 MIND-AID — Depression Text Classifier", "Baseline demo (TF‑IDF + LinearSVC)"), unsafe_allow_html=True)

# Sidebar: samples and info
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

with st.sidebar:
    st.header("Try an example")
    examples = {
        "Everyday stress (normal)": "Had a long day at work but a jog helped clear my head."
        ,
        "Low mood (normal)": "Feeling a bit low this week, but talking to friends helps."
        ,
        "Depressive tone (flag)": "I can't find any reason to get out of bed and everything feels pointless."
        ,
        "Hopelessness (flag)": "I'm tired of pretending I'm okay. Nothing seems to matter anymore."
    }
    choice = st.selectbox("Examples", list(examples.keys()))
    if st.button("Use example", type="secondary"):
        st.session_state.user_text = examples[choice]
        st.toast("Example loaded", icon="✅")

    st.divider()
    st.caption("Models loaded from: ")
    st.code(MODEL_DIR, language="text")


col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Enter text")
    user_text = st.text_area(
        "Input text",
        key="user_text",
        height=220,
        placeholder="Paste a Reddit-style post or any short message...",
        label_visibility="collapsed",
    )
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
            st.info("This is a baseline demo. Do not use for clinical decisions.")
    else:
        st.write("Output will appear here after you click Predict.")

st.markdown("---")
st.caption("Tip: run `py src/train.py` first to (re)generate models.")
