# Phase 2 Transformer (Deferred)

All implementation code has been temporarily removed per project plan.
Files remain as placeholders to preserve structure, but the **output
layout is now aligned with Phase 1**:

- `output/models/` — where the fine-tuned transformer + tokenizer
	should be saved.
- `output/metrics/` — where evaluation summaries/metrics should be
	written.

## To Implement Later

- Tokenizer loading
- Model loading & fine-tuning
- Evaluation script
- Explanation utilities (SHAP or alternative)
- Demo Streamlit app

## Getting Started (When Ready)

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt

# Once implemented, this will fine-tune the transformer and
# save artifacts under ./output/models and metrics under
# ./output/metrics (see src/train_transformer.py):
python src/train_transformer.py
```

## Troubleshooting (Windows Git Bash)

- Ensure you activated the correct venv:

```bash
source .venv/Scripts/activate
python -c "import sys; print(sys.executable)"  # should point inside phase2_transformer/.venv
```

- VS Code: use "Python: Select Interpreter" and pick the interpreter under `phase2_transformer/.venv`.
- If you previously had a root venv, delete it (`rm -rf ../.venv`) and only use per-phase venvs.

## Notes

Do not commit partial experiments; implement fully then update this README.

## Folder Structure

- `src/` — Python modules (currently placeholders)
- `demo_app/` — Streamlit demo (placeholder)
- `output/models/` — expected location for transformer artifacts
- `output/metrics/` — expected location for evaluation metrics
- `tests/` — placeholder tests
- `requirements.txt` — phase-specific dependencies
