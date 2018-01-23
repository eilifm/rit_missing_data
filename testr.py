from generator import *
from shredder import *
from fitter import *
from fixer import *

import numpy as np


dist_types = [
    x_def_helper('uniform', coeff=5, low=0, high=100)
]

clean_fit_data, test_data = generate_ind_model(1, dist_types, intercept=10, n=100, noise_factor=.1)

wreck_results = []
for i in range(10):
    for i in np.arange(0, 1, .05):
        wrecked_data = clean_fit_data.copy(deep=True)

        if i != 0:
            uniform_shred_cols(['x1'], i, wrecked_data)
            wrecked_data = wrecked_data.dropna()
            # wrecked_data = fix_cols({'x1': 'mean'}, wrecked_data)

        w_fitted, w_metrics, = fit_lm(wrecked_data, test_data)

        wreck_results.append((i,
                              w_fitted.nobs,
                              w_metrics['r2'],
                              w_metrics['r2_adj'],
                              w_metrics['bic'],
                              w_metrics['beta_ci']['x1']['range'],
                              w_metrics['r2_pred'],
                              w_metrics['mse_pred']))


results_df = pd.DataFrame(wreck_results, columns=['pct_missing', 'nobs', 'r2', 'r2_adj','bic', 'beta_x1_rng', 'r2_pred', 'mse'])
print(results_df.groupby('pct_missing').mean())


# print(w_metrics['bic'], w_metrics['aic'])
# print(c_metrics['bic'], c_metrics['aic'])

