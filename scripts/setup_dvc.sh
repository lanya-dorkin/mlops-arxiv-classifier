#!/bin/bash
# Initialize DVC repository

set -e

echo "Initializing DVC..."
dvc init

echo "Configuring DVC..."
dvc config core.autostage true

echo "Setting up local remote storage..."
mkdir -p dvc_storage
dvc remote add -d myremote ./dvc_storage

echo "DVC initialized successfully!"
echo "To use cloud storage, run:"
echo "  dvc remote add -d myremote gdrive://YOUR_FOLDER_ID"
echo "or"
echo "  dvc remote add -d myremote s3://your-bucket/path"
