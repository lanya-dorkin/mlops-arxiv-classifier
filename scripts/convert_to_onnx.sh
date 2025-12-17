#!/bin/bash
# Convert model to ONNX

CHECKPOINT=${1:-"train_artifacts/model.pt"}
OUTPUT=${2:-"train_artifacts/model.onnx"}

echo "Converting $CHECKPOINT to ONNX format..."
python -m arxiv_classifier.commands export_onnx \
    --checkpoint_path="$CHECKPOINT" \
    --output_path="$OUTPUT"

echo "Done! ONNX model saved to $OUTPUT"
