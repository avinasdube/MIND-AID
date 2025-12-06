"""Visualization helpers (minimal)."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion(y_true, y_pred, title: str = "Confusion Matrix"):
	cm = confusion_matrix(y_true, y_pred)
	sns.heatmap(cm, annot=True, fmt="d")
	plt.title(title)
	plt.xlabel("Pred")
	plt.ylabel("True")
	return plt.gcf()
