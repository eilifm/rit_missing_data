import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random

def make_styles(n):
    p1 = random.choices(np.arange(6, 15, 2), k=n)
    p2 = random.choices(np.arange(6, 15, 2), k=n)
#    p3 = random.choices(np.arange(4, 20, 1), k=n)
#    p4 = random.choices(np.arange(4, 10, 1), k=n)
    return list(zip(p1, p2))#, p3, p4))

def gen1(datafile, outname, factor, responses=["rel_mse_pred", "r2"], x="pct_missing"):
    data = pd.read_csv("./report_data/"+datafile, index_col=0)

    if not isinstance(factor, list):
        factor = [factor]
    grouped = data.groupby(factor + [x]).mean().reset_index()

    if x == "pct_missing":
        grouped.loc[:, 'pct_missing'] = grouped.loc[:, 'pct_missing'] * 100

    grouped.set_index(x, inplace=True)

    factor_levels = grouped[factor[0]].unique()

    num_responses = len(responses)
    num_levels = len(factor_levels)

    f, axarr = plt.subplots(num_responses, sharex=True, figsize=(12, 7))
    linestyles = make_styles(num_levels)
    for plot_ix in range(num_responses):
        for level_ix in range(num_levels):
            axarr[plot_ix].plot(
                grouped.loc[
                    grouped[factor[0]] == factor_levels[level_ix],
                    responses[plot_ix]
                ], label=factor_levels[level_ix], dashes=linestyles[level_ix])

        if len(factor) > 1:
            label = "(" + ", ".join(factor) + ")"
        else:
            label = ",".join(factor)

        axarr[plot_ix].set_title('{} vs. {} for Levels of {}'.format(responses[plot_ix], x, label), fontsize=14)
        axarr[plot_ix].set_ylabel(responses[plot_ix])
        axarr[plot_ix].legend(shadow=True, fancybox=True)

    axarr[num_responses-1].set_xlabel(x)
    f.savefig('figures/'+outname, bbox_inches='tight', dpi=600)


def gen2(datafile, outname, factor, responses=["rel_mse_pred", "r2"], x="pct_missing"):
    data = pd.read_csv("./report_data/"+datafile, index_col=0)

    if not isinstance(factor, list):
        factor = [factor]
    grouped = data.groupby(factor + [x]).mean().reset_index()

    if x == "pct_missing":
        grouped.loc[:, 'pct_missing'] = grouped.loc[:, 'pct_missing'] * 100

    grouped.set_index(x, inplace=True)

    factor_levels = grouped[factor[0]].unique()

    num_responses = len(responses)
    num_levels = len(factor_levels)

    f, axarr = plt.subplots(num_responses, sharex=True, figsize=(12, 7))
    linestyles = make_styles(num_levels)
    for plot_ix in range(num_responses):
        for level_ix in range(num_levels):
            axarr[plot_ix].plot(
                grouped.loc[
                    grouped[factor[0]] == factor_levels[level_ix],
                    responses[plot_ix]
                ], label=factor_levels[level_ix], dashes=linestyles[level_ix])

        if len(factor) > 1:
            label = "(" + ", ".join(factor) + ")"
        else:
            label = ",".join(factor)

        axarr[plot_ix].set_title('{} vs. {} for Levels of {}'.format(responses[plot_ix], x, label), fontsize=14)
        axarr[plot_ix].set_ylabel(responses[plot_ix], fontsize=12)
        axarr[plot_ix].legend(shadow=True, fancybox=True)

    axarr[num_responses-1].set_xlabel(x, fontsize=12)
    f.savefig('figures/'+outname, bbox_inches='tight', dpi=600)


def gen3(path, subset_cols, subset_vals, factor, response, outname, dpi, pct=False):
    # Load data
    data = pd.read_csv("./report_data/"+path, index_col=0)

    # Type checks on factor
    if not isinstance(factor, list):
        factor = [factor]

    # SubSelect
    for scol_ix in range(len(subset_cols)):
        data = data.loc[data[subset_cols[scol_ix]].isin(subset_vals[scol_ix]), :]

    # Group
    grouped = data.groupby(factor + ['pct_missing']).mean().reset_index()
    grouped.loc[:, 'pct_missing'] = grouped.loc[:, 'pct_missing'] * 100
    grouped.set_index('pct_missing', inplace=True)

    if pct:
        grouped.loc[:, response] = grouped.loc[:, response] * 100

    # Plot it!
    linestyles = ('o', '^', 'v', '<', '>', '8', 's', 'p')
    factor_levels = grouped[factor[0]].unique()
    num_levels = len(factor_levels)
    f, ax = plt.subplots(1, figsize=(12, 7))
    for level_ix in range(num_levels):
        ax.plot(
            grouped.loc[grouped[factor[0]] == factor_levels[level_ix], response],
            linestyles[level_ix],
            label=factor_levels[level_ix],
        )

    if len(factor) > 1:
        label = "(" + ", ".join(factor) + ")"
    else:
        label = ",".join(factor)

    ax.set_title('{} vs. {} for Levels of {}'.format(response, '% Missing', label), fontsize=14)
    ax.legend(shadow=True, fancybox=True)
    ax.set_ylabel(response, fontsize=12)
    ax.set_xlabel('% of Data Missing ', fontsize=12)
    f.savefig('figures/' + outname, bbox_inches='tight', dpi=dpi)



# Bias Problems Discussion
# 100 Replications
# No Mean - All action_types
gen3("drop-invert-mean_beta12-10_x101_1.csv",
     ['action_type', 'sigma'],
     [
         ['invert', 'drop'],
         [1, 2]
     ],
     'action_type',
     'x1_bias_pct',
     'drop_invert_bias_demo1.png', 150, pct=True)


# Bias Problems Discussion
# 100 Replications
# No Mean - All action_types
gen3("drop-invert-mean_beta12-10_x101_2.csv",
     ['action_type', 'sigma'],
     [
         ['invert', 'drop'],
         [1, 2]
     ],
     'action_type',
     'x1_bias_pct',
     'drop_invert_bias_demo2.png', 150, pct=True)


# Bias Problems Discussion
# 500 Replications
# No Mean - All action_types
gen3("drop-invert-mean_beta12-10_x501_1.csv",
     ['action_type', 'sigma'],
     [
         ['invert', 'drop'],
         [1, 2]
     ],
     'action_type',
     'x1_bias_pct',
     'drop_invert_bias_demo3.png', 150, pct=True)


# Bias Problems Discussion
# 5 Replications
# No Mean - All action_types
gen3("drop-invert-mean_beta12-10_x5_1.csv",
     ['action_type', 'sigma'],
     [
         ['invert', 'drop'],
         [1, 2]
     ],
     'action_type',
     'x1_bias_pct',
     'drop_invert_bias_demo_low_rep.png', 150, pct=True)



gen1("drop-invert-mean_beta12-10_x501_1.csv", "ci_rng_demo", ["action_type"], responses=["x1_bias", "x1_ci_rng"])
print(make_styles(3))