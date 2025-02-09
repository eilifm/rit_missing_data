from generator import *
from shredder import *
from fitter import *
from fixer import *
from collections import OrderedDict
from sortedcontainers import SortedDict

def run(data_gen_dict, action_type, beta_sigma, sample_size, incr, lower_pct, upper_pct, rep_num):

    fit_data, test_data, true_coeffs = generate_ind_model(
                                            data_gen_dict["dist_list"],
                                            data_gen_dict["main_coeffs"],
                                            data_gen_dict["interaction_coeffs"],
                                            data_gen_dict["interactions"],
                                            intercept=10,
                                            n=sample_size,
                                            beta_sigma=beta_sigma
                                        )

    param_cols = []
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
        results_rec = [pct,
                            w_fitted.nobs,
                            w_metrics['r2'],
                            w_metrics['r2_adj'],
                            w_metrics['bic'],
                            w_metrics['r2_pred'],
                            w_metrics['mse_pred'],
                            w_metrics['mape_pred']
        ]

        # A TON OF FANCY FOOTWORK TO KEEP ALL THE PARAMS IN ORDER!
        param_results = SortedDict()
        for x_var in w_fitted.params.index:
            param_results[x_var] = SortedDict()
            param_results[x_var]['ci_rng'] = w_metrics['beta_ci'][x_var]['range']
            param_results[x_var]['in_target'] = b_estimate_results.loc[x_var, :].values[0]
            param_results[x_var]['estimate'] = w_fitted.params[x_var]
            param_results[x_var]['true_beta'] = true_coeffs[x_var]

        # A TON OF FANCY FOOTWORK TO KEEP ALL THE PARAMS IN ORDER!
        param_info = []
        for x_var in param_results.keys():
            for info in param_results[x_var].keys():
                param_info.append(param_results[x_var][info])
                if x_var + "_" +info not in param_cols:
                    param_cols.append(x_var + "_" +info)

        run_results.append(results_rec+param_info)

    results_cols = ['pct_missing',
     'nobs',
     'r2',
     'r2_adj',
     'bic',
     'r2_pred',
     'mse_pred',
     'mape_pred']+param_cols

    # # Load the results into a Pandas Dataframe
    results = pd.DataFrame(run_results, columns=results_cols)

    # results_agg = results.copy()
    # results_agg = results_agg.groupby('pct_missing').mean()

    # results_agg.loc[:, 'action_type'] = action_type
    results.loc[:, 'action_type'] = action_type
    results.loc[:, 'beta_sigma'] = beta_sigma
    results.loc[:, 'sample_size'] = sample_size
    results.loc[:, 'beta_x2/beta_x1'] = true_coeffs['x2']/true_coeffs['x1']
    results.loc[:, 'beta_x1:x2/beta_x1'] = true_coeffs['x1:x2']/true_coeffs['x1']


    return results


