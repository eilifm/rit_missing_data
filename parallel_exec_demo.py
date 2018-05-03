from generator import *
from shredder import *
from fitter import *
import datetime
from data_collection import upload, download
from fixer import *
from exec_tools import run
from sklearn.preprocessing import MinMaxScaler
import json
from joblib import Parallel, delayed
from math import sqrt
import time
import itertools
import argparse
import numpy as np

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    print(args.config)
    download(args.config+".json")

    with open(args.config+".json", 'r') as config:
        levels = json.load(config)

    return levels['levels'], args.config

if __name__ == "__main__":

    levels, config = parse()

    print(len(levels))
    #results = Parallel(n_jobs=-1, verbose=10)(delayed(run)(*args) for args in itertools.product(*levels))
#    np.random.shuffle(levels)
    results = Parallel(n_jobs=-1, verbose=10)(delayed(run)(*args) for args in levels)

    numeric_levels = list(set(itertools.chain.from_iterable([result[1] for result in results])))

    data_results = pd.concat([result[0] for result in results])

    grp_cols = [
        "pct_missing",
        "x1_true_beta",
        "x1:x2_true_beta",
        "x2_true_beta",
        "const_true_beta",
        "action_type",
        "sigma",
        "sample_size",
        "targets"
    ]
    grouped_data = data_results #data_results.groupby(grp_cols).mean().reset_index()

    grouped_data = grouped_data.sample(frac=1).reset_index(drop=True)

#    data_results.loc[:, ["cod_"+factor for factor in numeric_levels]] = scaler.transform(data_results.loc[:, numeric_levels])

    import time
    now = str(datetime.datetime.now().isoformat()).replace("/", "-").replace(" ", '_').replace(":", '-')
    now = str(int(time.time()*1000))
    outfile_name = config+"_"+now
    grouped_data.to_csv(outfile_name+".csv")

    scaler = MinMaxScaler()
    scaler.fit(grouped_data.loc[:, grp_cols]._get_numeric_data())
    numerics = grouped_data.loc[:, grp_cols]._get_numeric_data().columns

    grouped_data.loc[:, numerics] = scaler.transform(grouped_data.loc[:, numerics])

    grouped_data.to_csv(outfile_name+"_coded.csv")


    # with open(config + now+".json", 'w') as json_out:
    #     json.dump(list(itertools.product(*levels)), json_out, indent=4)
    upload(outfile_name+".csv")
    upload(outfile_name+"_coded.csv")

    #
    # print(time.time() - start)
