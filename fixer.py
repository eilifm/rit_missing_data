import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
from generator import *
from shredder import *
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


def inverse_fit_impute_interaction(x_name, y_name, df: pd.DataFrame):
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


def olsinv_singlex(input_df, target, verbose=False, test_mode=False):
    """
    
    Usage Example
    -------------
    
    >>> data_gen_dict = config_maker(2, 5, [.4], [[1,2]], [5])
    >>> fit_data, test_data, true_coeffs = generate_ind_model(
    ...                                    data_gen_dict["dist_list"],
    ...                                    data_gen_dict["main_coeffs"],
    ...                                    data_gen_dict["interaction_coeffs"],
    ...                                    data_gen_dict["interactions"],
    ...                                    intercept=1,
    ...                                    n=100,
    ...                                    beta_sigma=0 # Intentionally zero error to demonstrate invert operation
    ...                                )
    >>> fit_data.columns.values
    array(['x1', 'x2', 'x1:x2', 'y'], dtype=object)
    >>> tmp_data = fit_data.copy()
    >>> inverted,  fitted = olsinv_singlex(tmp_data, 'x1', test_mode=True)    
    >>> (np.round(tmp_data.loc[inverted.index, 'x1'].values, 10) == np.round(inverted.values, 10)).all()
    True
    
 
    Data Structure Requirements
    ---------------------------
    Data must follow R variable naming. Each variable must be uniquely named with the interaction noted with a `:`
    between each participating variable. 
    
    For example:
    >>> pd.DataFrame(
    ...     np.array([
    ...         [1, 2, 2, 18],
    ...         [2, 4, 8, 56]
    ...     ]),columns=['x1', 'x2', 'x1:x2', 'y'])
       x1  x2  x1:x2   y
    0   1   2      2  18
    1   2   4      8  56

    
    
    
    Parameters
    ----------
    input_df
    target
    verbose
    test_mode

    Returns
    -------

    """

    input_df = input_df.copy()

    # Build the model matrix from all the columns that are not "y" -- the response
    if test_mode:
        X_to_inv = input_df.loc[:, input_df.columns.values != 'y'].sample(
            int(input_df.shape[0] * .1), random_state=2)  # FOR TESTING
    else:
        X_to_inv = input_df.loc[
            input_df.loc[:,
            ~input_df.columns.isin(['x1'])
            ].isnull().sum(1) == 0,
            input_df.columns.values != 'y'
        ]

    # Error out if, by mistake, there is no missing data in the target column

    # Add in the constant column to the model matrix since it is not ordinarily part of the dataframe
    X_to_inv = sm.add_constant(X_to_inv)

    # Select the "y" -- response values coresponding to the rows with missing data
    if test_mode:
        y = input_df.loc[X_to_inv.index, 'y']  # FOR TESTING
    else:
        y = input_df.loc[X_to_inv.index, 'y']

    # Start by performing the standard OLS fit rows with no NaN values in ANY colums
    if test_mode:
        X_init = input_df.loc[~input_df.index.isin(X_to_inv.index), input_df.columns.values != 'y']
    else:
        X_init = input_df.loc[~input_df.isnull().any(1), input_df.columns.values != 'y']

    X_init = sm.add_constant(X_init)
    y_init = input_df.loc[X_init.index, 'y']

    # Fit the model to complete observations.
    inv_mod = sm.OLS(y_init, X_init).fit()

    if verbose:
        print("Shape of data for Init model: {}".format(str(X_init.shape)))
    # Generate a list of all interaction terms with the target column
    interact_cols = X_to_inv.columns[
        (X_to_inv.columns.str.contains(target + ":")) | (X_to_inv.columns.str.contains(":" + target))].values

    if verbose:
        print("Interaction Columns: {}".format(str(interact_cols)))

    # Find the main regressor names for the interaction factors
    # Eg. If the target for inverse computation is 'x1' and 'x1' has the interactions
    # 'x1:x2' and 'x1:x3' then this would yield the list ['x2', 'x3'] as we have "factored out" 'x1'
    m2_factored_cols = [re.sub('{}:|:{}'.format(target, target), '', inter) for inter in interact_cols]

    if verbose:
        print("Factored Column Names: {}".format(str(m2_factored_cols)))

    # Gather the [Beta] matrix for all columns that are not the target or an
    # interaction term that includes the target
    m1_betas = inv_mod.params.loc[
        (inv_mod.params.index != target) &
        ~(
            (inv_mod.params.index.str.contains(":" + target)) |
            (inv_mod.params.index.str.contains(target + ":"))
        )
        ]
    if verbose:
        print("Matrix 1 Beta Estimates: {}".format(m1_betas))

    # Gather the X values for all columns that are not the target or interaction terms
    m1_X_to_inv = X_to_inv.loc[:,
                  (
                      (X_to_inv.columns.values != target) &
                      ~(
                          X_to_inv.columns.str.contains(target + ":") |
                          X_to_inv.columns.str.contains(":" + target)
                      )
                  )
                  ]

    # Compute [X][Beta] for non-interaction terms
    # Eg. If 'x1' is target with one additional x and their two-way interaction
    # [X] = [1, 'x2']
    m1 = np.matmul(m1_X_to_inv, m1_betas)

    # Gather the [X] matrix for the factored interaction terms
    # Eg. If the target for inverse computation is 'x1' and 'x1' has the interactions
    # 'x1:x2' and 'x1:x3' then this factoring would yield ['x1'][1, 'x2', 'x3'] as we have "factored out" 'x1'
    m2_X_to_inv = np.append(
        np.ones((X_to_inv.shape[0], 1)),
        X_to_inv.loc[:, m2_factored_cols],
        axis=1
    )

    # Gather [Beta] matrix for the target and its participating interactions
    m2_betas = inv_mod.params.loc[(inv_mod.params.index == target) |
                                  (inv_mod.params.index.str.contains(target + ":")) |
                                  (inv_mod.params.index.str.contains(":" + target))]

    if verbose:
        print("Matrix 2 Beta Estimates: \n{} \n".format(m2_betas))

    # Compute [X][Beta] for interaction terms
    m2 = np.matmul(m2_X_to_inv, m2_betas.values)

    # Solve the inverted OLS equation using the m1 and m2 matrix components
    inverted_target = (y - m1) / m2

    return inverted_target, inv_mod


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