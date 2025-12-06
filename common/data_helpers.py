"""Data loading / splitting helpers."""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: str):
	return pd.read_csv(path)


def split_text_label(df, text_col="clean_text", label_col="is_depression", test_size=0.15):
	X = df[text_col].astype(str).values
	y = df[label_col].values
	return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
