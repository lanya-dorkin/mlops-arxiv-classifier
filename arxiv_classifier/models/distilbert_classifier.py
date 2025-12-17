"""DistilBERT classifier using PyTorch Lightning."""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import DistilBertModel

from arxiv_classifier.utils.metrics import (
    compute_accuracy,
    compute_macro_f1,
)


class DistilBertClassifier(pl.LightningModule):
    """DistilBERT-based text classifier using PyTorch Lightning."""

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        warmup_steps: int = 500,
        total_steps: Optional[int] = None,
    ):
        """Initialize classifier.

        Args:
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            dropout: Dropout rate
            freeze_backbone: Whether to freeze DistilBERT backbone
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps (for scheduler)
        """
        super().__init__()
        self.save_hyperparameters()

        self.backbone = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Logits
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[0][:, 0, :]  # [CLS] token
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            Loss value
        """
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])

        preds = torch.argmax(logits, dim=1)
        accuracy = compute_accuracy(preds.cpu().numpy(), batch["labels"].cpu().numpy())
        f1 = compute_macro_f1(preds.cpu().numpy(), batch["labels"].cpu().numpy())

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", accuracy, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])

        preds = torch.argmax(logits, dim=1)
        accuracy = compute_accuracy(preds.cpu().numpy(), batch["labels"].cpu().numpy())
        f1 = compute_macro_f1(preds.cpu().numpy(), batch["labels"].cpu().numpy())

        self.log("test/loss", loss)
        self.log("test/accuracy", accuracy)
        self.log("test/f1", f1)

    def configure_optimizers(self):
        """Configure optimizer and scheduler.

        Returns:
            Dictionary with optimizer and scheduler config
        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        if self.total_steps is None:
            return optimizer

        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
