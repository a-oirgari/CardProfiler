import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import FEATURES, RANDOM_STATE


def make_clean(df):
    df = df.drop_duplicates()
    df = df.drop(columns=["CUST_ID"])
    df = df.dropna()
    return df.reset_index(drop=True)

def prepare_clustering(df_clean):
    X = df_clean[FEATURES].copy()
    assert (X >= 0).all().all(), "log1p impose des valeurs >= 0"
    df_log = np.log1p(X)
    scaler = StandardScaler()
    df_prepare = pd.DataFrame(scaler.fit_transform(df_log), columns=FEATURES, index=df_clean.index)
    return df_log, df_prepare


