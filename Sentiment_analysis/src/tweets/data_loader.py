from Sentiment_analysis.src.logger.train_logger import get_logger
import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = get_logger()

def load_data(
    data_path: str,
    dataset_columns: list,
    dataset_encoding: str,
    engine: str,
    sentiment_col: str,
    text_col: str,
    test_size: float = 0.2,      
    random_state: int = 42 
):
    """
    Loads sentiment dataset, cleans columns,
    replaces sentiment values, and splits into train/test sets.
    """

    # Load dataset
    dataset = pd.read_csv(
        data_path,
        encoding=dataset_encoding,
        names=dataset_columns,
        engine=engine
    )

    # Keep only the needed columns
    dataset = dataset[[sentiment_col, text_col]]

    # Replace sentiment labels
    dataset[sentiment_col] = dataset[sentiment_col].replace(4, 1)

    # Train/Test Split
    train_df, test_df = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[sentiment_col]
    )

    LOGGER.info(f"Loaded dataset from: {data_path}")
    LOGGER.info(f"Dataset shape: {dataset.shape}")
    LOGGER.info(f"Train split: {train_df.shape}")
    LOGGER.info(f"Test split: {test_df.shape}")

    return train_df, test_df
