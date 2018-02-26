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

    data = {
        "dist_list": [
            x_def_helper('uniform', low=0, high=1),
            x_def_helper('uniform', low=0, high=1)
        ],
        "main_coeffs": [10, 5],
        "interactions": [
            [1, 2]
        ],
        "interaction_coeffs": [2]
    }

    data2 = {
        "dist_list": [
            x_def_helper('uniform', low=0, high=1),
            x_def_helper('uniform', low=0, high=1)
        ],
        "main_coeffs": [10, 2],
        "interactions": [
            [1, 2]
        ],
        "interaction_coeffs": [2]
    }

    # start = time.time()
    # results = Parallel(n_jobs=2)(delayed(run_new)() for i in range(100))

    levels = [
        (data,data2,),
        ("mean", "invert", "drop"),
        (.1,3),
        (20, 50, ),
        (.05,),
        (0,),
        (.8,),
        range(10)
    ]

    print(len(list(itertools.product(*levels))))
    # start = time.time()

    import pprint

    # runs = list(itertools.product(*levels))
    #
    # pprint.pprint(runs)
    #
    # print(run(*runs[0]))

    results = Parallel(n_jobs=-1, verbose=1)(delayed(run)(*args) for args in itertools.product(*levels))
    # results = pd.concat(results)
    #
    # print(results.shape)
    #
    # print(time.time() - start)
