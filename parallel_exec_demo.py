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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    print(args.config)
    download(args.config+".json")

    with open(args.config+".json", 'r') as config:
        levels = json.load(config)

    print(len(levels))
    #results = Parallel(n_jobs=-1, verbose=10)(delayed(run)(*args) for args in itertools.product(*levels))
    results = Parallel(n_jobs=-1, verbose=10)(delayed(run)(*args) for args in levels)

    numeric_levels = list(set(itertools.chain.from_iterable([result[1] for result in results])))

    data_results = pd.concat([result[0] for result in results])

    scaler = MinMaxScaler()

    scaler.fit(data_results.loc[:, numeric_levels])

    data_results = data_results.reindex(columns=data_results.columns.tolist() + ["cod_" + factor for factor in numeric_levels])
    data_results.loc[:, ["cod_"+factor for factor in numeric_levels]] = scaler.transform(data_results.loc[:, numeric_levels])

    now = str(datetime.datetime.now()).replace("/", "-").replace(" ", '_').replace(":", '-')
    outfile_name = now+".csv"
    data_results.to_csv(outfile_name)

    with open(args.config + now+".json", 'w') as json_out:
        json.dump(list(itertools.product(*levels)), json_out, indent=4)
    upload(outfile_name)

    #
    # print(time.time() - start)
