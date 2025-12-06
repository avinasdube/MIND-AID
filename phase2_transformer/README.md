# Phase 2 Transformer (Deferred)

All implementation code has been temporarily removed per project plan.
Files remain as placeholders to preserve structure.

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
# Add training command here once implemented
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
- `tests/` — placeholder tests
- `requirements.txt` — phase-specific dependencies
