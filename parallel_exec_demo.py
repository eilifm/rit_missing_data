from generator import *
from shredder import *
from fitter import *
from fixer import *
from exec_tools import run
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    from joblib import Parallel, delayed
    from math import sqrt
    import time
    import itertools

    maker_levels = [
        (2,),
        (10,),
        ([.05], [.5]), # beta_x2/beta_x1
        ([[1, 2]],), # Declare interactions
        ([1], [5]) # Levels of interaction coeff
    ]

    # start = time.time()
    # results = Parallel(n_jobs=2)(delayed(run_new)() for i in range(100))

    levels = [
        [config_maker(*args) for args in itertools.product(*maker_levels)],
        #("mean", "invert", "drop"),
        ("invert", "drop"),
        # (.1, .3),
        (.1,),
        (50, 100),
        (.05,),
        (0,),
        (.2,),
        range(100)
    ]

    print(len(list(itertools.product(*levels))))
    # start = time.time()

    runs = list(itertools.product(*levels))
    #
    #

    # print(run(*runs[0]))

    results = Parallel(n_jobs=-1, verbose=10)(delayed(run)(*args) for args in itertools.product(*levels))

    results = pd.concat(results)
    import datetime


    scaler = MinMaxScaler()

    scaler.fit(results[['beta_sigma', 'sample_size', 'beta_x2/beta_x1', 'pct_missing']])

    results[['_beta_sigma', '_sample_size', '_beta_x2/beta_x1', '_beta_x1:x2/beta_x1']] = pd.DataFrame([[0,0,0,0]], index=results.index)
    results.loc[:, ['_beta_sigma', '_sample_size', '_beta_x2/beta_x1', '_beta_x1:x2/beta_x1']] = scaler.transform(results.loc[:, ['beta_sigma', 'sample_size', 'beta_x2/beta_x1', 'beta_x1:x2/beta_x1']])

    results.to_csv(str(datetime.datetime.now()).replace("/", "-")+".csv")


    print(results.shape)
    #
    # print(time.time() - start)
