"""Basic logging configuration utility shared across phases."""

import logging
import os


def setup_logging(level: str = "INFO"):
	numeric = getattr(logging, level.upper(), logging.INFO)
	logging.basicConfig(
		level=numeric,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)
	logging.getLogger("transformers").setLevel(logging.WARNING)
	return logging.getLogger("mindaid")
