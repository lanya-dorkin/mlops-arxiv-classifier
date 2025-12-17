"""Data downloading utilities."""

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def download_data(data_dir: str = "./data") -> Path:
    """Download ArXiv dataset from Kaggle.

    Args:
        data_dir: Directory to save data to

    Returns:
        Path to downloaded data directory
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    dataset_id = "sumitm004/arxiv-scientific-research-papers-dataset"
    api.dataset_download_files(dataset_id, path=data_dir, unzip=True)

    return data_dir
