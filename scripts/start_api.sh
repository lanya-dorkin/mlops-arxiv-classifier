#!/bin/bash
# Start FastAPI server

CHECKPOINT=${1:-"train_artifacts/model.pt"}
PORT=${2:-8000}

echo "Starting API server on port $PORT..."
python -m arxiv_classifier.commands serve \
    --checkpoint_path="$CHECKPOINT" \
    --port="$PORT"
