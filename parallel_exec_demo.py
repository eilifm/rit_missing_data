from generator import *
from shredder import *
from fitter import *
from fixer import *
from exec_tools import run

if __name__ == "__main__":
    from joblib import Parallel, delayed
    from math import sqrt
    import time
    import itertools

    maker_levels = [
        (2,),
        (10,),
        ([.1], [.5], [1]),
        ([[1, 2]],),
        ([1], [5], [10])
    ]

    # start = time.time()
    # results = Parallel(n_jobs=2)(delayed(run_new)() for i in range(100))

    levels = [
        [config_maker(*args) for args in itertools.product(*maker_levels)],
        ("mean", "invert", "drop"),
        (.1, .3),
        (20, 50, ),
        (.05,),
        (0,),
        (.8,),
        range(10)
    ]

    print(len(list(itertools.product(*levels))))
    # start = time.time()

    runs = list(itertools.product(*levels))
    #
    #

    # print(run(*runs[0]))

    results = Parallel(n_jobs=-2, verbose=1)(delayed(run)(*args) for args in itertools.product(*levels))

    results = pd.concat(results)
    import datetime
    results.to_csv(str(datetime.datetime.now())+".csv")

    #
    # print(results.shape)
    #
    # print(time.time() - start)
