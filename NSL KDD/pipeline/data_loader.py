import pandas as pd
import os


def load_nsl_kdd_data(data_dir="../data/BinaryClassify"):
    """
    Load the NSL-KDD dataset from the specified directory.

    Parameters:
    data_dir (str): Directory where the encoded CSV files are stored.

    Returns:
    tuple: (train_data, test_data) as pandas DataFrames
    """
    train_path = os.path.join(data_dir, "train_nsl_kdd_binary_encoded.csv")
    test_path = os.path.join(data_dir, "test_nsl_kdd_binary_encoded.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Train or test data file not found. Please ensure the files exist in the specified directory.")

    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    return train_data, test_data
