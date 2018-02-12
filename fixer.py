import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error


def fix_cols(col_fix_methods:dict, df: pd.DataFrame):
    """
    
    
    Parameters
    ----------
    col_fix_methods
    df

    Returns
    -------

    """
    fixes = {}

    for col, method in col_fix_methods.items():
        if method == 'median':
            fixes[col] = df.median()[col]
        elif method == 'mean':
            fixes[col] = df.mean()[col]
        else:

            raise ValueError("LOL")
    return df.fillna(value=fixes)


def inverse_fit_impute(x_name, y_name, df: pd.DataFrame):
    """
    
    Invert the assumed model of y = B0 + B1*X1 + e
    X1 = (y - B0)/B1
    
    - Pull out the missing data
    - Fit normal model of y = B0 + B1*X1
    - Use the fitted parameters to compute missing X values
    - Fill in missing X values
    - Return fixed Dataframe

    Parameters
    ----------
    x_name
    y_name
    df

    Returns
    -------

    """

    # Gather all the non-NULL rows in the Xs
    #X = df.loc[~df[x_name].isnull(), df.columns != y_name]

    X = df.loc[~df[x_name].isnull(), x_name]
    X = sm.add_constant(X)

    # Gather all of the corresponding Y values
    y = df.loc[~df[x_name].isnull(), 'y']

    # Define the model
    model = sm.OLS(y, X)

    # Collect results
    results = model.fit()

    # Ignore this for now
    # df.loc[df[x_name].isnull(), "was_null"] = True
    # df.loc[~df[x_name].isnull(), "was_null"] = False


    df.loc[df[x_name].isnull(), x_name] = np.divide(
        np.subtract(
            df.loc[df[x_name].isnull(), y_name],
            results.params['const']
        ),
        results.params[x_name])


    return df, results.params

def rand_replace(x_name, df: pd.DataFrame):
    """
    
    Parameters
    ----------
    x_name: Name of column to fill in.
    df: DataFrame with missing data.

    Returns
    -------

    """
    required_samples = len(df.loc[df[x_name].isnull(), x_name])

    # Randomly sample the remaining data with replacement and fill in the missing data
    df.loc[df[x_name].isnull(), x_name] = df.loc[~df[x_name].isnull(), x_name].sample(required_samples, replace=True).values

    return df