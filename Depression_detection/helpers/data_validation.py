import pandas as pd

def clean_and_validate_dataset(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Cleans and validates a text classification dataset.
    
    Steps:
    1. Drops duplicates.
    2. Removes empty or whitespace-only text.
    3. Drops rows with nulls in text or label columns.
    4. Keeps only labels 0 or 1.
    5. Prints class distribution and final shape.
    
    Args:
        df (pd.DataFrame): Input dataset.
        text_col (str): Name of the text column.
        label_col (str): Name of the label column.
        
    Returns:
        pd.DataFrame: Cleaned and validated dataset.
    """
    df = df.copy()
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Remove empty or whitespace-only text
    df = df[df[text_col].str.strip() != '']
    
    # Drop rows with nulls in text or label columns
    df.dropna(subset=[text_col, label_col], inplace=True)
    
    # Keep only labels 0 or 1
    df = df[df[label_col].isin([0, 1])]
    
    # Display class distribution and shape
    print("Class distribution:")
    print(df[label_col].value_counts())
    print("\nFinal dataset shape:", df.shape)
    
    return df
