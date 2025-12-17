"""PyTorch Dataset and DataModule for ArXiv data."""

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer

from arxiv_classifier.data.preprocessing import load_and_preprocess


class ArxivDataset(Dataset):
    """PyTorch Dataset for ArXiv papers."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: DistilBertTokenizer,
        max_length: int = 512,
    ):
        """Initialize dataset.

        Args:
            dataframe: DataFrame with 'title', 'summary', and 'category_encoded'
            tokenizer: DistilBertTokenizer instance
            max_length: Maximum sequence length
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        """Get item by index.

        Args:
            idx: Index

        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        row = self.dataframe.iloc[idx]
        title = str(row.get("title", ""))
        summary = str(row.get("summary", ""))

        text = f"{title} {summary}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(row["category_encoded"], dtype=torch.long),
        }


class ArxivDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for ArXiv dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4,
        train_date: str = "2023-01-01",
        val_date: str = "2024-01-01",
    ):
        """Initialize DataModule.

        Args:
            data_dir: Directory containing data
            batch_size: Batch size for dataloaders
            max_length: Maximum sequence length
            num_workers: Number of workers for dataloaders
            train_date: Cutoff date for train set
            val_date: Cutoff date for val set
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.train_date = train_date
        self.val_date = val_date

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.label_encoder = None

    def prepare_data(self) -> None:
        """Download data if needed."""
        pass

    def setup(self, stage: str = None) -> None:
        """Setup datasets for train/val/test.

        Args:
            stage: Stage name (fit, test, predict, None)
        """
        train_df, val_df, test_df, encoder = load_and_preprocess(
            self.data_dir, self.train_date, self.val_date
        )

        self.label_encoder = encoder

        self.train_dataset = ArxivDataset(train_df, self.tokenizer, self.max_length)
        self.val_dataset = ArxivDataset(val_df, self.tokenizer, self.max_length)
        self.test_dataset = ArxivDataset(test_df, self.tokenizer, self.max_length)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        if self.label_encoder is None:
            raise RuntimeError("setup() must be called before accessing num_classes")
        return len(self.label_encoder.classes_)

    @property
    def class_names(self) -> list[str]:
        """Return class names."""
        if self.label_encoder is None:
            raise RuntimeError("setup() must be called before accessing class_names")
        return list(self.label_encoder.classes_)
