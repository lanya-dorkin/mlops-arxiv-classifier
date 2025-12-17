"""Command-line interface using Fire and Hydra."""

import json
import pickle
from pathlib import Path
from typing import Optional

import fire
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from transformers import DistilBertTokenizer

from arxiv_classifier.data.dataset import ArxivDataModule
from arxiv_classifier.data.preprocessing import load_and_preprocess
from arxiv_classifier.models.baseline import BaselineModel
from arxiv_classifier.models.distilbert_classifier import DistilBertClassifier
from arxiv_classifier.utils.git import get_git_commit
from arxiv_classifier.utils.logger import get_logger, setup_logger
from arxiv_classifier.utils.mlflow import MLflowContext

logger = get_logger(__name__)


class Commands:
    """CLI commands for model training and evaluation."""

    @staticmethod
    def download():
        """Download dataset from Kaggle."""
        from arxiv_classifier.data.downloader import download_data

        logger.info("Starting dataset download from Kaggle...")
        data_dir = download_data()
        logger.info(f"Dataset download completed: {data_dir}")

    @staticmethod
    def train(config_name: str = "config"):
        """Train the main DistilBERT model.

        Args:
            config_name: Name of Hydra config file (without .yaml extension)
        """
        logger.info(f"Starting training with config: {config_name}")
        config_dir = Path(__file__).parent.parent / "configs"

        with initialize_config_dir(
            config_dir=str(config_dir.absolute()), version_base=None
        ):
            cfg = compose(config_name=config_name)

        logger.info(f"Configuration loaded: model={cfg.model.name}, seed={cfg.seed}")
        pl.seed_everything(cfg.seed)

        # Download data if needed
        from arxiv_classifier.data.downloader import download_data

        data_dir = Path(cfg.data.data_dir)
        if not data_dir.exists() or not list(data_dir.glob("*.csv")):
            logger.info("Data not found, downloading...")
            download_data(str(data_dir))

        # Setup MLflow (optional)
        with MLflowContext(
            cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name
        ) as mlflow_ctx:
            # Log parameters
            mlflow_ctx.log_params(
                {
                    "model": cfg.model.name,
                    "learning_rate": cfg.training.learning_rate,
                    "batch_size": cfg.data.batch_size,
                    "max_epochs": cfg.training.max_epochs,
                    "freeze_backbone": cfg.model.freeze_backbone,
                }
            )
            mlflow_ctx.log_param("git_commit", get_git_commit())

            # Setup data
            datamodule = ArxivDataModule(
                data_dir=cfg.data.data_dir,
                batch_size=cfg.data.batch_size,
                max_length=cfg.data.max_length,
                num_workers=cfg.data.num_workers,
                train_date=cfg.data.train_split_date,
                val_date=cfg.data.val_split_date,
            )
            datamodule.setup()

            # Setup model
            model = DistilBertClassifier(
                num_classes=datamodule.num_classes,
                learning_rate=cfg.training.learning_rate,
                dropout=cfg.model.dropout,
                freeze_backbone=cfg.model.freeze_backbone,
            )

            # Setup trainer
            mlflow_logger = None
            if mlflow_ctx.mlflow_available:
                try:
                    mlflow_logger = pl.loggers.MLFlowLogger(
                        experiment_name=cfg.mlflow.experiment_name,
                        tracking_uri=cfg.mlflow.tracking_uri,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create MLFlowLogger: {e}. Continuing without it."
                    )

            trainer = pl.Trainer(
                max_epochs=cfg.training.max_epochs,
                logger=mlflow_logger,
                accelerator="auto",
                devices="auto",
                callbacks=[
                    pl.callbacks.ModelCheckpoint(
                        monitor="val/f1",
                        mode="max",
                        save_top_k=1,
                    ),
                ],
            )

            # Train
            logger.info(f"Starting training for {cfg.training.max_epochs} epochs...")
            trainer.fit(model, datamodule=datamodule)
            logger.info("Training completed")

            # Test
            logger.info("Running test set evaluation...")
            trainer.test(model, datamodule=datamodule)
            logger.info("Test evaluation completed")

            # Save model and artifacts
            artifacts_dir = Path("train_artifacts")
            artifacts_dir.mkdir(exist_ok=True)

            model_path = artifacts_dir / "model.pt"
            trainer.save_checkpoint(model_path)
            mlflow_ctx.log_artifact(str(model_path))

            # Save label encoder
            encoder_path = artifacts_dir / "label_encoder.pkl"
            with open(encoder_path, "wb") as f:
                pickle.dump(datamodule.label_encoder, f)
            mlflow_ctx.log_artifact(str(encoder_path))

            logger.info(f"Model saved to {model_path}")
            logger.info(f"Label encoder saved to {encoder_path}")

    @staticmethod
    def baseline(config_name: str = "config"):
        """Train the TF-IDF baseline model.

        Args:
            config_name: Name of Hydra config file (without .yaml extension)
        """
        config_dir = Path(__file__).parent.parent / "configs"

        with initialize_config_dir(
            config_dir=str(config_dir.absolute()), version_base=None
        ):
            cfg = compose(config_name=config_name, overrides=["model=baseline"])

        # Download data if needed
        from arxiv_classifier.data.downloader import download_data

        data_dir = Path(cfg.data.data_dir)
        if not data_dir.exists() or not list(data_dir.glob("*.csv")):
            logger.info("Data not found, downloading...")
            download_data(str(data_dir))

        # Setup MLflow (optional)
        experiment_name = f"{cfg.mlflow.experiment_name}-baseline"
        with MLflowContext(cfg.mlflow.tracking_uri, experiment_name) as mlflow_ctx:
            mlflow_ctx.log_param("model_type", "baseline_tfidf")
            mlflow_ctx.log_param("git_commit", get_git_commit())

            # Load data
            logger.info("Loading and preprocessing data...")
            train_df, val_df, test_df, encoder = load_and_preprocess(
                data_dir,
                cfg.data.train_split_date,
                cfg.data.val_split_date,
            )

            # Prepare texts
            train_texts = (
                train_df["title"].astype(str) + " " + train_df["summary"].astype(str)
            ).tolist()
            val_texts = (
                val_df["title"].astype(str) + " " + val_df["summary"].astype(str)
            ).tolist()
            test_texts = (
                test_df["title"].astype(str) + " " + test_df["summary"].astype(str)
            ).tolist()

            # Train baseline
            logger.info(
                f"Training baseline model with max_features={cfg.model.max_features}..."
            )
            baseline = BaselineModel(max_features=cfg.model.max_features)
            baseline.train(train_texts, train_df["category_encoded"].values)
            logger.info("Baseline model training completed")

            # Evaluate
            logger.info("Evaluating on validation set...")
            val_metrics = baseline.evaluate(
                val_texts, val_df["category_encoded"].values, encoder.classes_
            )
            logger.info(
                f"Validation metrics - Macro F1: {val_metrics['macro_f1']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}"
            )

            logger.info("Evaluating on test set...")
            test_metrics = baseline.evaluate(
                test_texts, test_df["category_encoded"].values, encoder.classes_
            )
            logger.info(
                f"Test metrics - Macro F1: {test_metrics['macro_f1']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}"
            )

            # Log metrics
            mlflow_ctx.log_metric("val_macro_f1", val_metrics["macro_f1"])
            mlflow_ctx.log_metric("val_accuracy", val_metrics["accuracy"])
            mlflow_ctx.log_metric("test_macro_f1", test_metrics["macro_f1"])
            mlflow_ctx.log_metric("test_accuracy", test_metrics["accuracy"])

            # Log per-class metrics
            for class_name, metrics in test_metrics["per_class"].items():
                # Sanitize class name for MLflow (handled in log_metric, but explicit here for clarity)
                mlflow_ctx.log_metric(f"test_{class_name}_f1", metrics["f1"])
                mlflow_ctx.log_metric(
                    f"test_{class_name}_precision", metrics["precision"]
                )

            # Save model
            artifacts_dir = Path("train_artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            model_path = artifacts_dir / "baseline_model.pkl"
            baseline.save(model_path)
            mlflow_ctx.log_artifact(str(model_path))

            logger.info(f"Baseline model saved to {model_path}")

    @staticmethod
    def test(checkpoint_path: str, config_name: str = "config"):
        """Test a trained model.

        Args:
            checkpoint_path: Path to model checkpoint
            config_name: Name of Hydra config file (without .yaml extension)
        """
        logger.info(f"Starting model testing with checkpoint: {checkpoint_path}")
        config_dir = Path(__file__).parent.parent / "configs"

        with initialize_config_dir(
            config_dir=str(config_dir.absolute()), version_base=None
        ):
            cfg = compose(config_name=config_name)

        # Setup data
        logger.info("Setting up data module...")
        datamodule = ArxivDataModule(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            max_length=cfg.data.max_length,
            num_workers=cfg.data.num_workers,
            train_date=cfg.data.train_split_date,
            val_date=cfg.data.val_split_date,
        )
        datamodule.setup()

        # Load model
        logger.info(f"Loading model from {checkpoint_path}...")
        model = DistilBertClassifier.load_from_checkpoint(checkpoint_path)
        logger.info("Model loaded successfully")

        # Test
        logger.info("Running test evaluation...")
        trainer = pl.Trainer(accelerator="auto", devices="auto")
        trainer.test(model, datamodule=datamodule)
        logger.info("Test evaluation completed")

    @staticmethod
    def infer(
        checkpoint_path: str = "train_artifacts/model.pt",
        title: Optional[str] = None,
        summary: Optional[str] = None,
        json_file: Optional[str] = None,
        max_length: int = 512,
    ):
        """Run inference on new data.

        Args:
            checkpoint_path: Path to model checkpoint
            title: Paper title (if single prediction)
            summary: Paper summary (if single prediction)
            json_file: Path to JSON file with multiple samples
            max_length: Maximum sequence length (default: 512)
        """
        logger.info(f"Starting inference with checkpoint: {checkpoint_path}")

        # Load model
        logger.info("Loading model and tokenizer...")
        model = DistilBertClassifier.load_from_checkpoint(checkpoint_path)
        model.eval()
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # Load label encoder
        encoder_path = Path("train_artifacts/label_encoder.pkl")
        label_encoder = None
        if encoder_path.exists():
            with open(encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
            logger.info(f"Label encoder loaded: {len(label_encoder.classes_)} classes")
        else:
            logger.warning("Label encoder not found, using class indices")

        # Prepare input
        if json_file:
            logger.info(f"Loading samples from {json_file}...")
            with open(json_file) as f:
                data = json.load(f)
            samples = data if isinstance(data, list) else [data]
            logger.info(f"Loaded {len(samples)} samples")
        elif title and summary:
            samples = [{"title": title, "summary": summary}]
            logger.info("Processing single sample")
        else:
            raise ValueError("Provide either title+summary or json_file")

        results = []
        for i, sample in enumerate(samples, 1):
            text = f"{sample['title']} {sample['summary']}"
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits = model(encoding["input_ids"], encoding["attention_mask"])
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)

            # Get category name
            if label_encoder:
                category = label_encoder.inverse_transform([pred_idx.item()])[0]
            else:
                category = f"class_{pred_idx.item()}"

            result = {
                "title": sample["title"][:50] + "..."
                if len(sample["title"]) > 50
                else sample["title"],
                "category": category,
                "confidence": float(confidence.item()),
            }
            results.append(result)

            logger.info(
                f"Sample {i}/{len(samples)} - Category: {result['category']}, "
                f"Confidence: {result['confidence']:.4f}"
            )

        logger.info(f"Inference completed for {len(results)} samples")
        return results

    @staticmethod
    def export_onnx(
        checkpoint_path: str = "train_artifacts/model.pt",
        output_path: str = "train_artifacts/model.onnx",
    ):
        """Export model to ONNX format.

        Args:
            checkpoint_path: Path to PyTorch checkpoint
            output_path: Output path for ONNX model
        """
        logger.info(f"Exporting model to ONNX: {checkpoint_path} -> {output_path}")
        from arxiv_classifier.utils.export import export_to_onnx

        export_to_onnx(checkpoint_path, output_path)
        logger.info(f"ONNX model exported to {output_path}")

    @staticmethod
    def export_torchscript(
        checkpoint_path: str = "train_artifacts/model.pt",
        output_path: str = "train_artifacts/model.pt.script",
    ):
        """Export model to TorchScript format.

        Args:
            checkpoint_path: Path to PyTorch checkpoint
            output_path: Output path for TorchScript model
        """
        logger.info(
            f"Exporting model to TorchScript: {checkpoint_path} -> {output_path}"
        )
        from arxiv_classifier.utils.export import export_to_torchscript

        export_to_torchscript(checkpoint_path, output_path)
        logger.info(f"TorchScript model exported to {output_path}")

    @staticmethod
    def serve(
        checkpoint_path: str = "train_artifacts/model.pt",
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """Start FastAPI server for inference.

        Args:
            checkpoint_path: Path to model checkpoint
            host: Host to bind to
            port: Port to bind to
        """
        import uvicorn

        from arxiv_classifier.api import app, load_model

        logger.info(f"Loading model from {checkpoint_path}...")
        load_model(checkpoint_path)
        logger.info("Model loaded successfully")

        logger.info(f"Starting API server on {host}:{port}")
        logger.info(f"API Documentation available at http://{host}:{port}/docs")
        uvicorn.run(app, host=host, port=port)


def main():
    """Main entry point."""
    setup_logger()
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
