import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import FEATURES, RANDOM_STATE


def make_clean(df):
    df = df.drop_duplicates()
    df = df.drop(columns=["CUST_ID"])
    df = df.dropna()
    return df.reset_index(drop=True)




