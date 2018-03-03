import numpy as np
import pandas as pd
import scipy.stats as stats
from sortedcontainers import SortedDict
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def make_fit_metrics(fittedmodel):
    """
    
    Parameters
    ----------
    fittedmodel

    Returns
    -------

    """
    ci = fittedmodel.conf_int(.05).to_dict(orient='index')
    m_ci = {}
    for var, vals in ci.items():
        m_ci[var] = {'low': vals[0], 'high': vals[1], 'range': vals[1]- vals[0]}

    metrics = {
        "pvalues": fittedmodel.pvalues.to_dict(),
        "r2_adj": fittedmodel.rsquared_adj,
        "r2": fittedmodel.rsquared,
        'bic': fittedmodel.bic,
        'aic': fittedmodel.aic,
        'beta_ci': m_ci
    }
    return metrics


def fit_lm(df, test_df, alpha=.05):
    """
    
    Parameters
    ----------
    df
    alpha

    Returns
    -------

    """

    X = df.loc[:, df.columns != "y"]
    test_X = test_df.loc[:, df.columns != "y"]

    y = df.loc[:, 'y']
    test_y = test_df.loc[:, 'y']

    X = sm.add_constant(X)
    test_X = sm.add_constant(test_X)

    model = sm.OLS(y, X)
    results = model.fit()

    f_metrics = make_fit_metrics(results)

    y_pred = results.predict(test_X)

    pred_metrics = {
        'r2_pred': r2_score(test_y, y_pred),
        'mse_pred': mean_squared_error(test_y, y_pred),
        'mape_pred': mean_absolute_percentage_error(test_y, y_pred)
    }

    f_metrics.update(**pred_metrics)

    return results, f_metrics


def beta_target_check(metrics, coeffs, as_dataframe=False):
    beta_target_results = SortedDict()
    for key in coeffs:
        if metrics['beta_ci'][key]['low'] <= coeffs[key] <= metrics['beta_ci'][key]['high']:
            beta_target_results[key]=SortedDict({'in_ci': 1})
        else:
            beta_target_results[key] = SortedDict({'in_ci': 0})

    return beta_target_results