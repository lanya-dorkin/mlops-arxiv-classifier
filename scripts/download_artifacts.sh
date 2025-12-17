#!/bin/bash
# Download and setup data

set -e

echo "Downloading data from Kaggle..."
python -m arxiv_classifier.commands download

echo "Updating DVC lock file..."
dvc commit download

echo "Committing DVC files..."
git add dvc.lock
git commit -m "Update data in DVC tracking"

echo "Done!"
