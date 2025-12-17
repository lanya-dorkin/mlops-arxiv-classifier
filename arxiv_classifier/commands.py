"""Command-line interface using Fire and Hydra."""

import json
import pickle
import subprocess
from pathlib import Path
from typing import Optional

import fire
import mlflow
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from transformers import DistilBertTokenizer

from arxiv_classifier.data.dataset import ArxivDataModule
from arxiv_classifier.data.downloader import download_data
from arxiv_classifier.data.preprocessing import load_and_preprocess
from arxiv_classifier.models.baseline import BaselineModel
from arxiv_classifier.models.distilbert_classifier import DistilBertClassifier


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def ensure_data_available(data_dir: Path) -> None:
    """Ensure data is available via DVC or download from Kaggle.

    Args:
        data_dir: Path to data directory
    """
    if data_dir.exists() and list(data_dir.glob("*.csv")):
        return

    print("Data not found. Attempting to pull from DVC...")
    try:
        import dvc.repo

        repo = dvc.repo.Repo()
        repo.pull()
        print("Data pulled from DVC successfully")
    except Exception as e:
        print(f"DVC pull failed ({e}), downloading from Kaggle...")
        download_data(str(data_dir))


class Commands:
    """CLI commands for model training and evaluation."""

    @staticmethod
    def download():
        """Download dataset from Kaggle."""
        print("Downloading ArXiv dataset...")
        data_dir = download_data()
        print(f"Dataset downloaded to {data_dir}")

    @staticmethod
    def train(config_name: str = "config"):
        """Train the main DistilBERT model.

        Args:
            config_name: Name of Hydra config file (without .yaml extension)
        """
        config_dir = Path(__file__).parent.parent / "configs"

        with initialize_config_dir(
            config_dir=str(config_dir.absolute()), version_base=None
        ):
            cfg = compose(config_name=config_name)

        # Set seed
        pl.seed_everything(cfg.seed)

        # Download data if needed
        data_dir = Path(cfg.data.data_dir)
        if not data_dir.exists() or not list(data_dir.glob("*.csv")):
            Commands.download()

        # Setup MLflow
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(
                {
                    "model": cfg.model.name,
                    "learning_rate": cfg.training.learning_rate,
                    "batch_size": cfg.data.batch_size,
                    "max_epochs": cfg.training.max_epochs,
                    "freeze_backbone": cfg.model.freeze_backbone,
                }
            )
            mlflow.log_param("git_commit", get_git_commit())

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
            logger = pl.loggers.MLFlowLogger(
                experiment_name=cfg.mlflow.experiment_name,
                tracking_uri=cfg.mlflow.tracking_uri,
            )

            trainer = pl.Trainer(
                max_epochs=cfg.training.max_epochs,
                logger=logger,
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
            print("Starting training...")
            trainer.fit(model, datamodule=datamodule)

            # Test
            print("Running test set evaluation...")
            trainer.test(model, datamodule=datamodule)

            # Save model and artifacts
            artifacts_dir = Path("train_artifacts")
            artifacts_dir.mkdir(exist_ok=True)

            model_path = artifacts_dir / "model.pt"
            trainer.save_checkpoint(model_path)
            mlflow.log_artifact(str(model_path))

            # Save label encoder
            encoder_path = artifacts_dir / "label_encoder.pkl"
            with open(encoder_path, "wb") as f:
                pickle.dump(datamodule.label_encoder, f)
            mlflow.log_artifact(str(encoder_path))

            print(f"Model saved to {model_path}")
            print(f"Label encoder saved to {encoder_path}")

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
        data_dir = Path(cfg.data.data_dir)
        if not data_dir.exists() or not list(data_dir.glob("*.csv")):
            Commands.download()

        # Setup MLflow
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(f"{cfg.mlflow.experiment_name}-baseline")

        with mlflow.start_run():
            mlflow.log_param("model_type", "baseline_tfidf")
            mlflow.log_param("git_commit", get_git_commit())

            # Load data
            print("Loading and preprocessing data...")
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
            print("Training baseline model...")
            baseline = BaselineModel(max_features=cfg.model.max_features)
            baseline.train(train_texts, train_df["category_encoded"].values)

            # Evaluate
            print("Evaluating on validation set...")
            val_metrics = baseline.evaluate(
                val_texts, val_df["category_encoded"].values, encoder.classes_
            )
            print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")

            print("Evaluating on test set...")
            test_metrics = baseline.evaluate(
                test_texts, test_df["category_encoded"].values, encoder.classes_
            )
            print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")

            # Log metrics
            mlflow.log_metric("val_macro_f1", val_metrics["macro_f1"])
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"])
            mlflow.log_metric("test_macro_f1", test_metrics["macro_f1"])
            mlflow.log_metric("test_accuracy", test_metrics["accuracy"])

            # Log per-class metrics
            for class_name, metrics in test_metrics["per_class"].items():
                mlflow.log_metric(f"test_{class_name}_f1", metrics["f1"])
                mlflow.log_metric(f"test_{class_name}_precision", metrics["precision"])

            # Save model
            artifacts_dir = Path("train_artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            model_path = artifacts_dir / "baseline_model.pkl"
            baseline.save(model_path)
            mlflow.log_artifact(str(model_path))

            print(f"Baseline model saved to {model_path}")

    @staticmethod
    def test(checkpoint_path: str, config_name: str = "config"):
        """Test a trained model.

        Args:
            checkpoint_path: Path to model checkpoint
            config_name: Name of Hydra config file (without .yaml extension)
        """
        config_dir = Path(__file__).parent.parent / "configs"

        with initialize_config_dir(
            config_dir=str(config_dir.absolute()), version_base=None
        ):
            cfg = compose(config_name=config_name)

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

        # Load model
        model = DistilBertClassifier.load_from_checkpoint(checkpoint_path)

        # Test
        trainer = pl.Trainer(accelerator="auto", devices="auto")
        trainer.test(model, datamodule=datamodule)

    @staticmethod
    def infer(
        checkpoint_path: str = "train_artifacts/model.pt",
        title: Optional[str] = None,
        summary: Optional[str] = None,
        json_file: Optional[str] = None,
    ):
        """Run inference on new data.

        Args:
            checkpoint_path: Path to model checkpoint
            title: Paper title (if single prediction)
            summary: Paper summary (if single prediction)
            json_file: Path to JSON file with multiple samples
        """
        config_dir = Path(__file__).parent.parent / "configs"

        with initialize_config_dir(config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

        # Load model
        model = DistilBertClassifier.load_from_checkpoint(checkpoint_path)
        model.eval()

        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # Load label encoder
        encoder_path = Path("train_artifacts/label_encoder.pkl")
        label_encoder = None
        if encoder_path.exists():
            with open(encoder_path, "rb") as f:
                label_encoder = pickle.load(f)

        # Prepare input
        if json_file:
            with open(json_file) as f:
                data = json.load(f)
            samples = data if isinstance(data, list) else [data]
        elif title and summary:
            samples = [{"title": title, "summary": summary}]
        else:
            raise ValueError("Provide either title+summary or json_file")

        results = []
        for sample in samples:
            text = f"{sample['title']} {sample['summary']}"
            encoding = tokenizer(
                text,
                max_length=cfg.data.max_length,
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

            print(f"Title: {result['title']}")
            print(f"Category: {result['category']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print()

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
        from arxiv_classifier.utils.export import export_to_onnx

        export_to_onnx(checkpoint_path, output_path)

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
        from arxiv_classifier.utils.export import export_to_torchscript

        export_to_torchscript(checkpoint_path, output_path)

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

        # Load model
        load_model(checkpoint_path)

        # Start server
        print(f"Starting API server on {host}:{port}")
        print(f"API Documentation available at http://{host}:{port}/docs")
        uvicorn.run(app, host=host, port=port)


def main():
    """Main entry point."""
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
