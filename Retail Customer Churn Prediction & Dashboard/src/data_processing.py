# src/data_processing.py

import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file and parse date columns.
    """
    df = pd.read_csv(file_path, parse_dates=['signup_date'])
    return df
