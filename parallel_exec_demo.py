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

    data_gen = [x_def_helper('uniform', coeff=10, low=0, high=1)]

    # start = time.time()
    # results = Parallel(n_jobs=2)(delayed(run_new)() for i in range(100))

    levels = [
        (data_gen,),
        ("mean", "invert", "drop"),
        (.1, .3),
        (20, 50, 100),
        (.05,),
        (0,),
        (.8,),
        range(10)
    ]

    print(len(list(itertools.product(*levels))))
    start = time.time()
    results = Parallel(n_jobs=-1, verbose=1)(delayed(run)(*args) for args in itertools.product(*levels))

    results = pd.concat(results)

    print(results.shape)

    print(time.time() - start)
