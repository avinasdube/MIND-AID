# Using Phase 2 Transformer Results in Your Research Paper

This document explains how to regenerate and use all quantitative results and figures produced by the Phase 2 transformer model in a research paper.

## 1. Regenerating Phase 2 Results

1. Activate the Phase 2 virtual environment (if you use one).
2. From the `MIND-AID/phase2_transformer` directory, run:

   ```bash
   python src/train_transformer.py
   python src/evaluate_transformer.py
   ```

This will:
- Fine‑tune the transformer model and save it under `output/models/`.
- Evaluate the model on the full dataset and write metrics and figures under `output/metrics/`.

## 2. What Files Are Produced

After running `evaluate_transformer.py`, you should have:

- `output/metrics/eval_metrics_phase2.json`
- `output/metrics/eval_report_phase2.txt`
- `output/metrics/confusion_matrix_phase2.png`
- `output/metrics/roc_phase2.png`
- `output/metrics/pr_curve_phase2.png`

### 2.1. `eval_metrics_phase2.json`

This JSON file contains high-level numeric metrics, for example:

- `accuracy`: overall classification accuracy on the evaluation set.
- `n_samples`: number of examples evaluated.
- `confusion_matrix`: raw 2×2 confusion matrix values.
- `roc_auc`: Area Under the ROC Curve.

**How to use in a paper:**
- Report accuracy and ROC AUC in the main text or in a metrics table.
- Use `n_samples` to specify the evaluation set size.
- Optionally convert the `confusion_matrix` into percentages and include them in a results table.

### 2.2. `eval_report_phase2.txt`

This is the scikit‑learn classification report for the transformer model.

**How to use in a paper:**
- Extract per‑class precision, recall, and F1‑scores to build a comparison table (e.g., Phase 1 vs. Phase 2).
- Quote key improvements in text (e.g., “The transformer model improves F1 on the ‘depressed’ class from 0.xx to 0.yy”).

### 2.3. `confusion_matrix_phase2.png`

This figure shows a row‑normalised 2×2 confusion matrix for the transformer model:

- X‑axis: predicted labels (`Not depressed / Normal`, `Depressed / Needs attention`).
- Y‑axis: true labels.
- Each cell is coloured by the row‑normalised proportion using a colourful but ordered colormap.
- Each cell is annotated with:
  - The raw count.
  - The percentage within that true label row.
- The title includes the overall accuracy.

**How to use in a paper:**
- Include this as a figure to show how the transformer distributes its errors across classes.
- Suggested caption: “Confusion matrix for the Phase 2 transformer model, showing counts and row‑normalised percentages for each class.”
- You can directly compare it with the Phase 1 confusion matrix in the same figure panel or as separate figures.

### 2.4. `roc_phase2.png`

This figure shows the ROC curve for the transformer model.

**How to use in a paper:**
- Include this figure when discussing the improved discriminative power of the transformer relative to the baseline.
- Suggested caption: “ROC curve for the Phase 2 transformer model on the depression detection task.”
- Mention the ROC AUC from this figure or from `eval_metrics_phase2.json` in your text.

### 2.5. `pr_curve_phase2.png`

This figure shows the Precision–Recall curve for the transformer model.

**How to use in a paper:**
- Use this figure when focusing on positive‑class performance and imbalanced data.
- Suggested caption: “Precision–Recall curve for the Phase 2 transformer model.”
- You may report the area under the PR curve if you compute it separately, or discuss qualitative behaviour (e.g., “precision remains above 0.xx up to yy% recall”).

## 3. Building Tables From the Outputs

### 3.1. Overall metrics table (Phase 1 vs. Phase 2)

You can build a comparison table across phases using values from:

- `phase1_baseline/output/metrics/eval_metrics_phase1.json`
- `phase2_transformer/output/metrics/eval_metrics_phase2.json`

For example, create a table with:

- Columns: Model, Accuracy, ROC AUC.
- Rows: `Phase 1 – TF–IDF + LinearSVC`, `Phase 2 – Transformer`.

You can also leverage the existing comparison script in `scripts/` to generate a Markdown table and summary.

### 3.2. Per‑class performance table

From `eval_report_phase2.txt`, extract:

- Per‑class Precision, Recall, F1‑score, and Support.

You can present this alone (to focus on the transformer) or side‑by‑side with Phase 1’s per‑class metrics to highlight improvements.

## 4. Referencing the Figures in Your Paper

When using the Phase 2 figures:

- Refer to the confusion matrix when discussing how the transformer model behaves on different classes (e.g., reduction in false negatives).
- Refer to the ROC and PR curves when discussing overall ranking quality and trade‑offs between precision and recall.
- If you include both Phase 1 and Phase 2 curves in the same panel (optional), clearly label each curve in the legend and mention this in your caption.

## 5. Reproducibility Notes

- Always note the commit hash or version of the code when generating results.
- Indicate that you ran `python src/train_transformer.py` followed by `python src/evaluate_transformer.py` from the `phase2_transformer` directory.
- Record any hyperparameter settings and training details (e.g., base model name, epochs, batch size) alongside your reported metrics.
- Mention that the Phase 2 model uses a transformer‑based architecture (e.g., DistilBERT) fine‑tuned on the same depression dataset as Phase 1.
