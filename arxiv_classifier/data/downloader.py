"""Data downloading utilities."""

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from arxiv_classifier.utils.logger import get_logger

logger = get_logger(__name__)


def download_data(data_dir: str = "./data") -> Path:
    """Download ArXiv dataset from Kaggle.

    Args:
        data_dir: Directory to save data to

    Returns:
        Path to downloaded data directory
    """
    data_dir = Path(data_dir)
    logger.info(f"Preparing to download dataset to {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    dataset_id = "sumitm004/arxiv-scientific-research-papers-dataset"
    logger.info(f"Downloading dataset: {dataset_id}")
    api.dataset_download_files(dataset_id, path=data_dir, unzip=True)

    csv_files = list(data_dir.glob("*.csv"))
    logger.info(f"Download completed. Found {len(csv_files)} CSV file(s) in {data_dir}")
    return data_dir
