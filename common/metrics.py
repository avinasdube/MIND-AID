"""Shared metrics helpers."""

from sklearn.metrics import classification_report, accuracy_score


def summarize_classification(y_true, y_pred):
	return {
		"accuracy": accuracy_score(y_true, y_pred),
		"report": classification_report(y_true, y_pred, output_dict=True),
	}
