import itertools
from generator import *
import json
from data_collection import upload


maker_levels = [
    (2,),
    (10,),
    ([.2], [.3]), # beta_x2/beta_x1
    ([
        [1, 2]
     ],), # Declare interactions
    ([.0000001], [.5], [10]),  # Levels of interaction coeff
    ('uniform',)
]

gen_levels = [len(x) for x in maker_levels]

# start = time.time()
# results = Parallel(n_jobs=2)(delayed(run_new)() for i in range(100))

levels = [
    [config_maker(*args) for args in itertools.product(*maker_levels)],
    ("mean", "invert", "drop"),
#    (.1, .3),
        (.2,),
    (50, 100),  # Initial sample sized
#        (50,),  # Initial sample sized
    (.05,),
    (0,),  # Lower bound on percent missing data
    (.5,),  # Upper bound on percent missing data
    (['x1'], ['x2'], ['x1', 'x2']),  # Select which columns to shred
    list(range(1))
]

total_levels = gen_levels + [len(x) for x in levels[1::]]
total_levels = map(str, total_levels)
file_name = str("x".join(total_levels))+ ".json"

print(file_name)
with open(file_name, 'w') as json_out:
    json.dump(list(itertools.product(*levels)), json_out)
#
upload(file_name)

