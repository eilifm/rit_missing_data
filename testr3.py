from generator import *
from shredder import *
from fitter import *
from fixer import *
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)

def run():
    x1_uniform_clean_fit_data, x1_uniform_test_data, x1_uniform_coeffs = generate_ind_model(
                                                            [
                                                                x_def_helper('uniform', coeff=10, low=0, high=1),
                                                                x_def_helper('uniform', coeff=10, low=0, high=1)
                                                            ],
                                                            intercept=10,
                                                            n=1000,
                                                            beta_sigma=.2
                                                        )


    print(x1_uniform_clean_fit_data)

    # wrecked_data = x1_uniform_clean_fit_data.copy(deep=True)
    # #wrecked_data = uniform_shred_cols(['x1'], .5, wrecked_data)
    #
    # fixed_data, w_impute_coeff = inverse_fit_impute('x1', 'y', wrecked_data)
    # #fixed_data = fix_cols({'x1': 'mean'}, wrecked_data)
    # ex_fitted, ex_metrics, = fit_lm(fixed_data, x1_uniform_test_data)
    #
    #
    # return ex_fitted.rsquared





# from joblib import Parallel, delayed
# from math import sqrt
# import time
# start = time.time()
# results = Parallel(n_jobs=2)(delayed(run)() for i in range(100))
#
# print(time.time()-start)
#
#
# # print(w_metrics['bic'], w_metrics['aic'])
# # print(c_metrics['bic'], c_metrics['aic'])
#
