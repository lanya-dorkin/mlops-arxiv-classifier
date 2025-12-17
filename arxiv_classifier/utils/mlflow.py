"""MLflow utilities for optional tracking."""

import re

import mlflow

from arxiv_classifier.utils.logger import get_logger

logger = get_logger(__name__)


def sanitize_metric_name(name: str) -> str:
    """Sanitize metric name for MLflow compatibility.

    MLflow allows only: alphanumerics, underscores (_), dashes (-),
    periods (.), spaces ( ), colons (:) and slashes (/).

    Args:
        name: Original metric name

    Returns:
        Sanitized metric name
    """
    # Replace parentheses and commas with underscores (common in category names)
    sanitized = re.sub(r"[(),]", "_", name)
    # Replace spaces with underscores for consistency
    sanitized = re.sub(r"\s+", "_", sanitized)
    # Replace other invalid characters with underscores
    # Keep allowed: alphanumerics, _, -, ., :, /
    sanitized = re.sub(r"[^a-zA-Z0-9_\-.:/]", "_", sanitized)
    # Replace multiple consecutive underscores with single one
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


class MLflowContext:
    """Context manager for optional MLflow tracking."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        """Initialize MLflow context.

        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.mlflow_available = False
        self.run = None

    def __enter__(self):
        """Enter MLflow context."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self.run = mlflow.start_run()
            self.mlflow_available = True
            logger.info(f"MLflow tracking enabled: {self.tracking_uri}")
        except Exception as e:
            logger.warning(
                f"MLflow not available ({e}). Continuing without tracking. "
                "To enable tracking, start MLflow server: mlflow ui --port 8080"
            )
            self.mlflow_available = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit MLflow context."""
        if self.mlflow_available and self.run:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

    def log_param(self, key: str, value):
        """Log parameter to MLflow if available."""
        if self.mlflow_available:
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log MLflow parameter {key}: {e}")

    def log_params(self, params: dict):
        """Log parameters to MLflow if available."""
        if self.mlflow_available:
            try:
                mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Failed to log MLflow parameters: {e}")

    def log_metric(self, key: str, value: float):
        """Log metric to MLflow if available."""
        if self.mlflow_available:
            try:
                sanitized_key = sanitize_metric_name(key)
                mlflow.log_metric(sanitized_key, value)
            except Exception as e:
                logger.warning(f"Failed to log MLflow metric {key}: {e}")

    def log_artifact(self, artifact_path: str):
        """Log artifact to MLflow if available."""
        if self.mlflow_available:
            try:
                mlflow.log_artifact(artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log MLflow artifact {artifact_path}: {e}")
