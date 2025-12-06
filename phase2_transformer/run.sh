#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt || echo "No requirements.txt found"
echo "Phase 2 is deferred. Add training/eval commands here once implemented."
