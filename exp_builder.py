import itertools
from generator import *
import json
from data_collection import upload
import numpy


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

maker_levels = [
    (2,),
    (10,),
    ([.1], [1]), # beta_x2/beta_x1
    ([
        [1, 2]
     ],), # Declare interactions
    ([.000001], [10], [50]),  # Levels of interaction coeff
    ('uniform',)
]

gen_levels = [len(x) for x in maker_levels]

# start = time.time()
# results = Parallel(n_jobs=2)(delayed(run_new)() for i in range(100))

levels = [
    [config_maker(*args) for args in itertools.product(*maker_levels)],
    ("mean", "invert", "drop", "random"),
    (.1, .3),
    (50, 100, 200),  # Initial sample sized
    (.05,),
    (0,),  # Lower bound on percent missing data
    (.50,),  # Upper bound on percent missing data
    (['x1'], ['x2'], ['x1', 'x2']),  # Select which columns to shred
    list(range(5))
]

total_levels = gen_levels + [len(x) for x in levels[1::]]
total_levels = map(str, total_levels)
file_name = str("x".join(total_levels))+ ".json"

metadata = maker_levels + levels[1::]

levels = list(itertools.product(*levels))

out_data = {"metadata": metadata, "levels": levels}

with open(file_name, 'w') as json_out:
    json.dump(out_data, json_out, cls=MyEncoder)
#
print(file_name)
upload(file_name)

