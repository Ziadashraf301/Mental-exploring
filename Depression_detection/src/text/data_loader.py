from Depression_detection.src.logger.train_logger import get_logger
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from Depression_detection.src.utils.data_validation import clean_and_validate_dataset

LOGGER = get_logger()

def load_data(
    raw_data_path: str,
    target_column_names: list[str],
    text_column_names:list[str],
    labels_map: dict,
    test_size: float = 0.2,      
    random_state: int = 42 
):
    """
    Load and combine all CSV files inside a directory.
    Return train/test splits.
    """

    raw_data_path = Path(raw_data_path)

    if not raw_data_path.exists():
        LOGGER.error(f"Path does not exist: {raw_data_path}")
        raise ValueError(f"Path does not exist: {raw_data_path}")

    # List CSV files
    csv_files = list(raw_data_path.glob("*.csv"))

    if len(csv_files) == 0:
        LOGGER.error(f"No CSV files found in: {raw_data_path}")
        raise ValueError(f"No CSV files found in: {raw_data_path}")

    LOGGER.info(f"Found {len(csv_files)} CSV files")

    # Load & concatenate
    df_list = []
    for file in csv_files:
        LOGGER.info(f"Loading {file.name}")
        
        if file.name == 'Suicide_Detection.csv':
            df = pd.read_csv(file, encoding="ISO-8859-1")
        else:
            df = pd.read_csv(file)

        # Identify sentiment and text columns
        sentiment_col = None
        text_col = None

        for col in text_column_names:
            if col in df.columns:
                text_col = col
                break

        for col in target_column_names:
            if col in df.columns:
                sentiment_col = col
                break
        
        if sentiment_col is None or text_col is None:
            LOGGER.error(f"Required columns not found in {file.name}")
            raise ValueError(f"Required columns not found in {file.name}")
        
        df.rename(columns={sentiment_col: "is_depression", text_col: "filtered_tweet"}, inplace=True)
        
        df = df[["is_depression", "filtered_tweet"]]

        # Map sentiment labels
        df["is_depression"] = df["is_depression"].map(labels_map)

        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    df = clean_and_validate_dataset(df, "filtered_tweet", "is_depression")

    # Train/Test Split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["is_depression"]
    )

    LOGGER.info(f"Loaded dataset from: {raw_data_path}")
    LOGGER.info(f"Dataset shape: {df.shape}")
    LOGGER.info(f"Train split: {train_df.shape}")
    LOGGER.info(f"Test split: {test_df.shape}")

    return train_df, test_df


