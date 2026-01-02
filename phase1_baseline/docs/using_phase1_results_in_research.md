# Using Phase 1 Baseline Results in Your Research Paper

This document explains how to regenerate and use all quantitative results and figures produced by the Phase 1 baseline (TF–IDF + LinearSVC) in a research paper.

## 1. Regenerating Phase 1 Results

1. Activate the Phase 1 virtual environment (if you use one).
2. From the `MIND-AID/phase1_baseline` directory, run:

   ```bash
   python src/train.py
   python src/evaluate_baseline.py
   ```

This will:
- Train the baseline model (if not already trained) and save it under `output/models/`.
- Evaluate the model on the full dataset and write metrics and figures under `output/metrics/`.

## 2. What Files Are Produced

After running `evaluate_baseline.py`, you should have:

- `output/metrics/eval_metrics_phase1.json`
- `output/metrics/eval_report_phase1.txt`
- `output/metrics/confusion_matrix_phase1.png`
- `output/metrics/roc_phase1.png`
- `output/metrics/pr_curve_phase1.png`

### 2.1. `eval_metrics_phase1.json`

This JSON file contains high-level numeric metrics, for example:

- `accuracy`: overall classification accuracy on the evaluation set.
- `n_samples`: number of examples evaluated.
- `confusion_matrix`: raw 2×2 confusion matrix values.
- `roc_auc`: Area Under the ROC Curve.

**How to use in a paper:**
- Report accuracy and ROC AUC in the main text or in a metrics table.
- Use `n_samples` to clearly specify the evaluation set size.
- Optionally convert the `confusion_matrix` values into percentages and include them in a results table.

### 2.2. `eval_report_phase1.txt`

This is the scikit‑learn classification report (per‑class precision, recall, F1‑score, and support).

**How to use in a paper:**
- Extract the per‑class precision, recall, and F1‑scores to build a table (e.g., rows = classes, columns = metrics).
- You can quote key numbers directly in text (e.g., “The baseline achieved an F1‑score of 0.xx on the ‘depressed’ class”).

### 2.3. `confusion_matrix_phase1.png`

This figure shows a row‑normalised 2×2 confusion matrix:

- X‑axis: predicted labels (`Not depressed / Normal`, `Depressed / Needs attention`).
- Y‑axis: true labels.
- Each cell is coloured by the row‑normalised proportion and annotated with:
  - The raw count.
  - The percentage within that true label row.
- The title includes the overall accuracy.

**How to use in a paper:**
- Include this as a figure to illustrate error patterns.
- Suggested caption: “Confusion matrix for the Phase 1 TF–IDF + LinearSVC baseline, showing counts and row‑normalised percentages for each class.”
- You can refer to it as “Figure X” in your text and discuss, for example, false positives vs. false negatives.

### 2.4. `roc_phase1.png`

This figure shows the ROC curve:

- X‑axis: False Positive Rate.
- Y‑axis: True Positive Rate.
- The legend includes the ROC AUC value.

**How to use in a paper:**
- Include as a figure when discussing discriminative ability.
- Suggested caption: “ROC curve for the Phase 1 baseline model on the depression detection task.”
- Mention the ROC AUC from the legend or from `eval_metrics_phase1.json` in your text.

### 2.5. `pr_curve_phase1.png`

This figure shows the Precision–Recall curve:

- X‑axis: Recall.
- Y‑axis: Precision.

**How to use in a paper:**
- Include if you want to emphasise performance on the positive (depressed) class, especially when the data is imbalanced.
- Suggested caption: “Precision–Recall curve for the Phase 1 baseline model.”

## 3. Building Tables From the Outputs

### 3.1. Overall metrics table

You can combine values from `eval_metrics_phase1.json` and `eval_report_phase1.txt` into a concise table. For example:

- Columns: Accuracy, ROC AUC, Precision (Depressed), Recall (Depressed), F1 (Depressed).
- Row: “Phase 1 – TF–IDF + LinearSVC”.

### 3.2. Per‑class performance table

From `eval_report_phase1.txt`, extract per‑class precision, recall, and F1‑score and present them as:

- Rows: `Not depressed / Normal`, `Depressed / Needs attention`.
- Columns: Precision, Recall, F1‑score, Support.

## 4. Referencing the Figures in Your Paper

When writing the paper:

- Refer to the confusion matrix as a figure showing the distribution of correct and incorrect predictions per class.
- Refer to the ROC and PR curves when discussing threshold‑independent performance.
- Make sure to:
  - State the evaluation dataset (same as training CSV in `data/raw/depression_dataset_reddit_cleaned.csv`).
  - Mention that results come from the Phase 1 baseline model (TF–IDF + LinearSVC) implemented in this repository.

## 5. Reproducibility Notes

- Always note the commit hash or version of the code when you generate the results.
- Indicate that you ran `python src/train.py` followed by `python src/evaluate_baseline.py` from the `phase1_baseline` directory.
- If you modify any hyperparameters or data splits, record those changes alongside your reported numbers.
