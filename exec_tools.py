from generator import *
from shredder import *
from fitter import *
from fixer import *


def run(data_gen_list, action_type, beta_sigma, sample_size, incr, lower_pct, upper_pct, rep_num):


    fit_data, test_data, true_coeffs = generate_ind_model(
                                            len(data_gen_list),
                                            data_gen_list,
                                            intercept=1000,
                                            n=sample_size,
                                            beta_sigma=beta_sigma
                                        )

    run_results = []
    # Begin missing levels loop
    for pct in np.arange(lower_pct, upper_pct, incr):

        # Make a copy of the dataset so that we do not contaminate the memory
        wrecked_data = fit_data.copy(deep=True)

        # Shred the data unless pct == 0
        # Specify that we are shredding x1
        uniform_shred_cols(['x1'], pct, wrecked_data)

        if action_type == 'drop':
            fixed_data = wrecked_data.dropna()
        elif action_type == 'mean':
            fixed_data = fix_cols({'x1': 'mean'}, wrecked_data)
        elif action_type == 'invert':
            fixed_data, w_impute_coeff = inverse_fit_impute('x1', 'y', wrecked_data)
        elif action_type == 'random':
            fixed_data = rand_replace('x1', wrecked_data)
        else:
            raise ValueError("Invalid action_type: %s" % action_type)

        # Fit the model
        w_fitted, w_metrics, = fit_lm(fixed_data, test_data)

        # Check if beta estimates are in the CI
        b_estimate_results = beta_target_check(w_metrics, true_coeffs, as_dataframe=True)

        # Collect some results
        run_results.append((pct,
                            w_fitted.nobs,
                            w_metrics['r2'],
                            w_metrics['r2_adj'],
                            w_metrics['bic'],
                            w_metrics['beta_ci']['x1']['range'],
                            b_estimate_results.loc['x1', :].values[0],
                            w_fitted.params['x1'],
                            w_metrics['r2_pred'],
                            w_metrics['mse_pred']))

    # Load the results into a Pandas Dataframe
    results = pd.DataFrame(run_results, columns=
    ['pct_missing',
     'nobs',
     'r2',
     'r2_adj',
     'bic',
     'beta_x1_rng',
     'beta_x1_target',
     'beta_x1',
     'r2_pred',
     'mse'])

    results_agg = results.copy()
    results_agg = results_agg.groupby('pct_missing').mean()

    results_agg.loc[:, 'action_type'] = action_type
    results.loc[:, 'action_type'] = action_type
    results.loc[:, 'beta_sigma'] = beta_sigma
    results.loc[:, 'sample_size'] = sample_size

    return results


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from math import sqrt
    import time
    import itertools

    data_gen = [x_def_helper('uniform', coeff=10, low=0, high=1)]

    # start = time.time()
    # results = Parallel(n_jobs=2)(delayed(run_new)() for i in range(100))

    levels = [
        (data_gen,),
        ("mean", "invert", "drop"),
        (.1, .3, .5),
        (20, 50, 100, 200),
        (.05,),
        (0,),
        (.8,),
        range(10)
    ]

    print(len(list(itertools.product(*levels))))
    start = time.time()
    results = Parallel(n_jobs=-1, verbose=1)(delayed(run_new)(*args) for args in itertools.product(*levels))

    results = pd.concat(results)

    print(results.shape)

    print(time.time() - start)


    #print(run_new(data_gen, "mean", .1, 100,.05,0,.8).shape)
