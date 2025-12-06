# MIND-AID

Multi-phase research & demo project for early detection support of depression/anxiety signals in short text (and later multimodal inputs). Not for clinical decision making.

## Phases

- **Phase 1 (Baseline)**: TF-IDF + LinearSVC classical ML baseline.
- **Phase 2 (Transformer)**: DistilBERT fine-tuning + SHAP token-level explanations.
- **Phase 3 (Multimodal)**: Experimental fusion of text + facial/audio embeddings.

## Directory Overview

```text
phase1_baseline/       # Frozen classical baseline (reproducible)
phase2_transformer/    # Transformer development + SHAP
phase3_multimodal/     # Optional experimental fusion work
common/                # Shared utilities (metrics, logging, data helpers)
models/                # Artifacts grouped by phase
data/                  # raw/ and processed/ datasets
scripts/               # helper scripts (exports, release prep)
deployment/            # Dockerfile, k8s manifests
tests/                 # Integration / high-level tests
```bash

## Getting Started (Phase 1)

```bash
cd phase1_baseline
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py
```

## Running Phases Separately

- Phase 1: lives in `phase1_baseline/` (baseline TF‑IDF + LinearSVC). Artifacts: `models/phase1/`.
- Phase 2: lives in `phase2_transformer/` (deferred placeholders kept). Implement later.
- Phase 3: lives in `phase3_multimodal/` (deferred placeholders kept). Implement later.

Each phase has its own `requirements.txt`, `src/`, `tests/`, and (optional) `demo_app/`.

## Documentation

See ongoing notes & progress: `docs/progress_explainer.md`

## Disclaimer

Research prototype. Not a medical device. Do not rely on outputs for clinical decisions.
