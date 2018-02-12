from generator import *
from shredder import *
from fitter import *
from fixer import *

import numpy as np

np.random.seed(10)

dist_types = [
    x_def_helper('uniform', coeff=5, low=0, high=100)
]

clean_fit_data, test_data, coeffs = generate_ind_model(1, dist_types, intercept=100, n=100, noise_factor=.1)

wrecked_data = clean_fit_data.copy(deep=True)

# if i != 0:
uniform_shred_cols(['x1'], .5, wrecked_data)
print(wrecked_data)

rand_replace('x1', wrecked_data)

print(wrecked_data)

# wreck_results = []
#
# for repl in range(10):
#     for i in np.arange(0, 1, .05):
#         wrecked_data = clean_fit_data.copy(deep=True)
#
#         # if i != 0:
#         uniform_shred_cols(['x1'], i, wrecked_data)
#         wrecked_data, w_impute_coeff = inverse_fit_impute('x1', 'y', wrecked_data)
# #        print(i)
#
#         w_fitted, w_metrics, = fit_lm(wrecked_data, test_data)
#
#         wreck_results.append((i,
#                               w_fitted.nobs,
#                               w_metrics['r2'],
#                               w_metrics['r2_adj'],
#                               w_metrics['bic'],
#                               w_metrics['beta_ci']['x1']['range'],
#                               w_metrics['r2_pred'],
#                               w_metrics['mse_pred'],
#                               w_impute_coeff['const'],
#                               w_impute_coeff['x1']
#                               ))
#
#
# results_df = pd.DataFrame(wreck_results, columns=['pct_missing',
#                                                   'nobs',
#                                                   'r2',
#                                                   'r2_adj',
#                                                   'bic',
#                                                   'beta_x1_rng',
#                                                   'r2_pred',
#                                                   'mse',
#                                                   'B0',
#                                                   'B1'
#                                                   ]
#                           )
#
# print(results_df.groupby('pct_missing').mean())
#

# print(w_metrics['bic'], w_metrics['aic'])
# print(c_metrics['bic'], c_metrics['aic'])

