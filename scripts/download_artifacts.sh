#!/bin/bash
# Download and setup data

set -e

echo "Downloading data from Kaggle..."
python -m arxiv_classifier.commands download

echo "Adding to DVC..."
dvc add data/

echo "Committing DVC files..."
git add data.dvc .dvc/.gitignore
git commit -m "Add data to DVC tracking"

echo "Done!"
