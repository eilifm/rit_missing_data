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


def generate_ind_model(dist_types: list,
                       main_coeffs:list,
                       interaction_coeffs: list,
                       interactions=[],
                       test_set_size=.5,
                       intercept=10,
                       n=1000,
                       beta_sigma=1.0):
    """

    Parameters
    ----------
    dist_types
    main_coeffs
    interaction_coeffs
    interactions
    test_set_size
    intercept
    n
    beta_sigma

    Returns
    -------

    """

    data = pd.DataFrame()
    main_effects = len(dist_types)

    # Generate each regressor iteritively
    for i in range(main_effects):
        data.loc[:, 'x'+str(i+1)] = generate_xvar(dist_types[i]['dist_name'], size=int(n*(1+test_set_size)), **dist_types[i]['dist_params'])

    coeffs_ = main_coeffs + interaction_coeffs
    coeffs_dict = {}

    # Interactions
    if len(interactions) >= 1:
        for inter_ix in range(len(interactions)):
            inter_factors = list(map(lambda x: "x" + str(x), interactions[inter_ix]))

            # Build the name of the interaction
            inter_name = ":".join(inter_factors)

            coeffs_dict[inter_name] = interaction_coeffs[inter_ix]

            data.loc[:, inter_name] = data.loc[:, inter_factors].prod(axis=1)


        #print("x"+str(inter[0]) + ":" + "x"+str(inter[1]))
    dotted = np.dot(data, coeffs_) + intercept


    for i in range(main_effects):
        coeffs_dict['x'+str(i+1)] = main_coeffs[i]

    coeffs_dict['const'] = intercept

    data['y'] = dotted + np.random.normal(0, max(coeffs_)*beta_sigma, size=int(n*(1+test_set_size)))

    fit = data.loc[0:n-1, :].copy()
    test = data.loc[n::, :].copy()
    return fit, test, coeffs_dict, max(coeffs_)*beta_sigma


def x_def_helper(name, **kwargs):
    """
    
    
    Parameters
    ----------
    name
    coeff
    kwargs

    Returns
    -------


    """
    return {'dist_name': name, 'dist_params': kwargs}


def rel_coeff_manager(base: int, factor_weights: list):
    """
    
    Parameters
    ----------
    base
    factor_weights
    interactions

    Returns
    -------

    """

    main_coeffs = [base] + list(np.multiply(base, factor_weights))

    return main_coeffs

def config_maker(num_x, base, coeff_weights, interactions, interaction_coeffs, underlying):
    if len(coeff_weights) + 1 > num_x:
        raise ValueError("len(coeff_weights) + 1 must be equal to num_x")

    if underlying == "uniform":
        init_dict = {
            "dist_list": [x_def_helper('uniform', low=0, high=1) for x in range(num_x)],

            "main_coeffs": rel_coeff_manager(base, coeff_weights),
            "interactions": interactions,
            "interaction_coeffs": interaction_coeffs
        }

    else:
        raise NotImplementedError()


    return init_dict


