import numpy as np
import pandas as pd


def uniform_shred_col(col_name, pct_trash, df):
    df.loc[np.random.choice(np.arange(0, df.shape[0]), int(pct_trash*df.shape[0]), replace=False), col_name] = np.NaN
    return df

def uniform_shred_cols(cols, pct_trash, df):
    df.loc[np.random.choice(np.arange(0, df.shape[0]), int(pct_trash*df.shape[0]), replace=False), cols] = np.NaN
    return df
