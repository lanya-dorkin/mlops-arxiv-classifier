"""Model export utilities."""

import torch

from arxiv_classifier.models.distilbert_classifier import DistilBertClassifier
from arxiv_classifier.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 14,
):
    """Export model to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output path for ONNX model
        opset_version: ONNX opset version
    """
    logger.info(f"Loading model from {checkpoint_path}...")
    model = DistilBertClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    logger.info(f"Exporting to ONNX format (opset_version={opset_version})...")
    # Create dummy input
    batch_size = 1
    seq_length = 512
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    # Export
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
    )

    logger.info(f"ONNX model exported to {output_path}")


def export_to_torchscript(
    checkpoint_path: str,
    output_path: str,
):
    """Export model to TorchScript format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output path for TorchScript model
    """
    logger.info(f"Loading model from {checkpoint_path}...")
    model = DistilBertClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    logger.info("Tracing model for TorchScript export...")
    # Trace model
    batch_size = 1
    seq_length = 512
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    traced_model = torch.jit.trace(
        model,
        (dummy_input_ids, dummy_attention_mask),
    )

    # Save
    traced_model.save(output_path)

    logger.info(f"TorchScript model exported to {output_path}")
