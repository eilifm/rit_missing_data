import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA


def generate_xvar(dist: str, size=1000, **kwargs):
    """
    
    
    Parameters
    ----------
    dist
    size
    kwargs

    Returns
    -------

    """

    if dist == 'uniform':
        return np.random.uniform(size=size, **kwargs)
    elif dist == 'normal':
        return np.random.normal(size=size, **kwargs)
    elif dist == 'lognormal':
        return np.random.lognormal(size=size, **kwargs)
    else:
        raise ValueError("Invalid distribution %s" % dist)


def generate_ind_model(p: int, dist_types: list, test_set_size=.5, intercept=10, n=1000, noise_factor=.1):
    """
    
    
    Parameters
    ----------
    p
    dist_types
    coefs
    intercept
    n

    Returns
    -------

    """
    data = pd.DataFrame()
    if p != len(dist_types):
        raise ValueError("p, len(dist_types) and len(coeffd) must all be equal")

    # Generate each regressor iteritively
    for i in range(p):
        data.loc[:, 'x'+str(i+1)] = generate_xvar(dist_types[i]['dist_name'], size=int(n*(1+test_set_size)), **dist_types[i]['dist_params'])

    dotted = np.dot(data, [dist_types[i]['x_coeff'] for i in range(p)]) + intercept

    coeffs = {}
    for i in range(p):
        coeffs['x'+str(i+1)] = dist_types[i]['x_coeff']

    coeffs['const'] = intercept

    data['y'] = dotted + np.random.normal(0, noise_factor*(np.max(dotted)-np.min(dotted)), size=int(n*(1+test_set_size)))

    fit = data.loc[0:n-1, :].copy()
    test = data.loc[n::, :].copy()
    return fit, test, coeffs


def x_def_helper(name, coeff, **kwargs):
    """
    
    
    Parameters
    ----------
    name
    coeff
    kwargs

    Returns
    -------

    """
    return {'dist_name': name, 'dist_params': kwargs, 'x_coeff': coeff}

