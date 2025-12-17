"""TF-IDF + Logistic Regression baseline model."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from arxiv_classifier.utils.metrics import (
    compute_accuracy,
    compute_macro_f1,
    compute_per_class_metrics,
)


class BaselineModel:
    """TF-IDF + Logistic Regression baseline."""

    def __init__(self, max_features: int = 10000):
        """Initialize baseline model.

        Args:
            max_features: Maximum number of TF-IDF features
        """
        self.max_features = max_features
        self.model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=max_features, stop_words="english"),
                ),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

    def train(self, texts: list[str], labels: np.ndarray) -> None:
        """Train the baseline model.

        Args:
            texts: List of text samples (title + summary)
            labels: Array of labels
        """
        self.model.fit(texts, labels)

    def predict(self, texts: list[str]) -> np.ndarray:
        """Make predictions.

        Args:
            texts: List of text samples

        Returns:
            Predicted labels
        """
        return self.model.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            texts: List of text samples

        Returns:
            Prediction probabilities
        """
        return self.model.predict_proba(texts)

    def evaluate(
        self,
        texts: list[str],
        labels: np.ndarray,
        class_names: list[str],
    ) -> dict:
        """Evaluate model performance.

        Args:
            texts: List of text samples
            labels: Array of labels
            class_names: List of class names

        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(texts)

        accuracy = compute_accuracy(predictions, labels)
        macro_f1 = compute_macro_f1(predictions, labels)
        per_class = compute_per_class_metrics(predictions, labels, class_names)

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": per_class,
        }

    def save(self, filepath: Path) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save model to
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, filepath: Path) -> "BaselineModel":
        """Load model from disk.

        Args:
            filepath: Path to load model from

        Returns:
            Loaded BaselineModel instance
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        instance = cls()
        instance.model = model
        return instance
