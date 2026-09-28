import pandas as pd
from src.config import RAW_PATH


def load_raw_data(path=RAW_PATH):
    return pd.read_csv(path)