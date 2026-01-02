"""Compare Phase 1 and Phase 2 models and produce plots/tables.

Run this from the repo root (MIND-AID/) with the appropriate venv
active. It expects that:

- Phase 1 has been trained and evaluated:
    - models under phase1_baseline/output/models
    - metrics under phase1_baseline/output/metrics (run evaluate_baseline)
- Phase 2 has been trained and evaluated:
    - models under phase2_transformer/output/models
    - metrics under phase2_transformer/output/metrics (run evaluate_transformer)

Outputs:
- scripts/phase_comparison_metrics.json
- scripts/phase_comparison_table.md (Markdown table for papers)
- scripts/phase_accuracy_bar.png (simple accuracy bar chart)
"""

from __future__ import annotations

import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PHASE1_METRICS = os.path.join(
    REPO_ROOT, "phase1_baseline", "output", "metrics", "eval_metrics_phase1.json"
)
PHASE2_METRICS = os.path.join(
    REPO_ROOT, "phase2_transformer", "output", "metrics", "eval_metrics_phase2.json"
)
OUT_DIR = os.path.dirname(__file__)


def _load_metrics(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    m1 = _load_metrics(PHASE1_METRICS)
    m2 = _load_metrics(PHASE2_METRICS)

    acc1 = m1.get("accuracy")
    acc2 = m2.get("accuracy")

    summary = {
        "phase1_accuracy": acc1,
        "phase2_accuracy": acc2,
        "phase1_n_samples": m1.get("n_samples"),
        "phase2_n_samples": m2.get("n_samples"),
    }

    # Save JSON summary
    with open(
        os.path.join(OUT_DIR, "phase_comparison_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    # Markdown table
    table_lines = [
        "| Model  | Accuracy | N samples |",
        "|--------|----------|-----------|",
        f"| Phase 1 (TF-IDF + LinearSVC) | {acc1:.4f} | {m1.get('n_samples')} |",
        f"| Phase 2 (Transformer)        | {acc2:.4f} | {m2.get('n_samples')} |",
    ]
    with open(
        os.path.join(OUT_DIR, "phase_comparison_table.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(table_lines))

    # Accuracy bar chart with nicer styling
    plt.style.use("seaborn-v0_8")
    labels = ["Phase 1", "Phase 2"]
    accuracies = [acc1, acc2]

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(labels, accuracies, color=["#60a5fa", "#f97316"], width=0.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Phase 1 vs Phase 2 accuracy", fontsize=14)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase_accuracy_bar.png"), dpi=220)
    plt.close(fig)

    print("Saved:")
    print(" - scripts/phase_comparison_metrics.json")
    print(" - scripts/phase_comparison_table.md")
    print(" - scripts/phase_accuracy_bar.png")


if __name__ == "__main__":  # pragma: no cover
    main()
