import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

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
        m_ci[var] = {'min': vals[0], 'max': vals[1], 'range': vals[1]- vals[0]}

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
        'mse_pred': mean_squared_error(test_y, y_pred)
    }

    f_metrics.update(**pred_metrics)

    return results, f_metrics
