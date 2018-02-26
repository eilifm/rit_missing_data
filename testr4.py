from generator import *
from shredder import *
from fitter import *
from fixer import *

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

def maker():
    init_dict = {
        "dist_list": None,
        "main_coeffs": None,
        "interactions": None,
        "interaction_coeffs": None
    }

import pprint
pprint.pprint(data2)