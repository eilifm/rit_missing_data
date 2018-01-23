import numpy as np
import pandas as pd

def fix_cols(col_fix_methods:dict, df: pd.DataFrame):
    fixes = {}

    for col, method in col_fix_methods.items():
        if method == 'median':
            fixes[col] = df.median()[col]
        elif method == 'mean':
            fixes[col] = df.mean()[col]
        else:

            raise ValueError("LOL")
    return df.fillna(value=fixes)

