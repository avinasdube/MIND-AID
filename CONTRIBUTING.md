# Contributing Guidelines

## Phase Isolation

Each phase (baseline, transformer, multimodal) is self-contained:

- Independent `requirements.txt`
- Separate virtual environment recommended per phase (`python -m venv .venv` inside the phase folder).
- Artifacts saved under `models/phaseX/` subdirectories.

## Directory Conventions

- `phase1_baseline/src/train.py` writes `tfidf_vectorizer.pkl`, `svm_model.pkl`.
- `phase2_transformer/src/train_transformer.py` writes HuggingFace model + tokenizer to `models/phase2/distilbert_mindAid/`.
- `phase3_multimodal/src/train_fusion.py` currently dummy; future artifacts go to `models/phase3/`.

## Tests

Minimal smoke tests exist in each phase `tests/` directory. Expand with real assertions as code matures.

## Logging & Utilities

Shared helpers live in `common/`. Avoid phase-specific logic there.

## Style

Follow PEP8; keep functions small and composable. Avoid breaking the frozen Phase 1 code beyond necessary path adjustments.

## CI

Add new tests or lint steps by editing `.github/workflows/ci.yml` (to be added) with a matrix over phase folders.

## Issues / PRs

Reference the phase (e.g., `Phase2:` prefix) in commit and PR titles.
