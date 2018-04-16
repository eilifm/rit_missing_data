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

        axarr[plot_ix].set_title('{} vs. {} for Levels of {}'.format(responses[plot_ix], x, factor[0]), fontsize=14)
        axarr[plot_ix].set_ylabel(responses[plot_ix])
        axarr[plot_ix].legend(shadow=True, fancybox=True)

    axarr[num_responses-1].set_xlabel(x)
    f.savefig('figures/'+outname, bbox_inches='tight')

gen1("drop_1x1x3x1x5x1x1x2x2x1x1x1x1x10_1523564702834.csv", "lol1.png", "x1:x2_true_beta")
print(make_styles(3))