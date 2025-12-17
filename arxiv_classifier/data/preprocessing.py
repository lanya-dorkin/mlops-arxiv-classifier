"""Data preprocessing utilities."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def split_by_date(
    df: pd.DataFrame,
    train_date: str,
    val_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by published_date.

    Args:
        df: Input dataframe with published_date column
        train_date: Cutoff date for train set (str format YYYY-MM-DD)
        val_date: Cutoff date for val set (str format YYYY-MM-DD)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df["published_date"] = pd.to_datetime(df["published_date"])
    train_date_dt = pd.to_datetime(train_date)
    val_date_dt = pd.to_datetime(val_date)

    train_df = df[df["published_date"] < train_date_dt].copy()
    val_df = df[
        (df["published_date"] >= train_date_dt) & (df["published_date"] < val_date_dt)
    ].copy()
    test_df = df[df["published_date"] >= val_date_dt].copy()

    return train_df, val_df, test_df


def encode_labels(categories: pd.Series) -> tuple[LabelEncoder, pd.Series]:
    """Encode categorical labels to integers.

    Args:
        categories: Series of category labels

    Returns:
        Tuple of (label_encoder, encoded_labels)
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(categories)
    return encoder, encoded


def load_and_preprocess(
    data_dir: Path,
    train_date: str = "2023-01-01",
    val_date: str = "2024-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """Load and preprocess ArXiv dataset.

    Args:
        data_dir: Directory containing downloaded data
        train_date: Cutoff date for train set
        val_date: Cutoff date for val set

    Returns:
        Tuple of (train_df, val_df, test_df, label_encoder)
    """
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df = pd.read_csv(csv_files[0])

    # Handle missing summaries
    df["summary"] = df.get("summary", "").fillna("")
    df["title"] = df.get("title", "").fillna("")

    # Remove duplicates by title (keep first)
    df = df.drop_duplicates(subset=["title"], keep="first")

    # Encode labels
    encoder, _ = encode_labels(df["category"])
    df["category_encoded"] = encoder.transform(df["category"])

    # Split by date
    train_df, val_df, test_df = split_by_date(df, train_date, val_date)

    return train_df, val_df, test_df, encoder
