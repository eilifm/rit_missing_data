from generator import *
from shredder import *
from fitter import *
import datetime
from data_collection import upload
from fixer import *
from exec_tools import run
from sklearn.preprocessing import MinMaxScaler
import json

if __name__ == "__main__":
    from joblib import Parallel, delayed
    from math import sqrt
    import time
    import itertools

    maker_levels = [
        (2,),
        (10,),
        ([.2], [.3], [.6]), # beta_x2/beta_x1
        ([
            [1, 2]
         ],), # Declare interactions
        ([.0000001], [.1], [.5], [1], [3], [5], [10]),  # Levels of interaction coeff
        ('uniform',)
    ]

    # start = time.time()
    # results = Parallel(n_jobs=2)(delayed(run_new)() for i in range(100))

    levels = [
        [config_maker(*args) for args in itertools.product(*maker_levels)],
        ("mean", "invert", "drop"),
        (.1, .2, .3),
#        (.1,),
        (50, 100, 500),  # Initial sample sized
#        (50,),  # Initial sample sized
        (.05,),
        (0,),  # Lower bound on percent missing data
        (.6,),  # Upper bound on percent missing data
        (['x1'], ['x2'], ['x1', 'x2']),  # Select which columns to shred
        list(range(2))
    ]

    print(len(list(itertools.product(*levels))))
    # start = time.time()

    runs = list(itertools.product(*levels))

    results = Parallel(n_jobs=-1, verbose=10)(delayed(run)(*args) for args in itertools.product(*levels))

    numeric_levels = list(set(itertools.chain.from_iterable([result[1] for result in results])))

    data_results = pd.concat([result[0] for result in results])

    scaler = MinMaxScaler()

    scaler.fit(data_results.loc[:, numeric_levels])

    data_results = data_results.reindex(columns=data_results.columns.tolist() + ["cod_" + factor for factor in numeric_levels])
    data_results.loc[:, ["cod_"+factor for factor in numeric_levels]] = scaler.transform(data_results.loc[:, numeric_levels])

    now = str(datetime.datetime.now()).replace("/", "-").replace(" ", '_').replace(":", '-')
    outfile_name = now+".csv"
    data_results.to_csv(outfile_name)
    with open(now+".json", 'w') as json_out:
        json.dump(list(itertools.product(*levels)), json_out, indent=4)
    upload(outfile_name)
    upload(now+".json")

    print(data_results.shape)
    #
    # print(time.time() - start)
