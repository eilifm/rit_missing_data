import numpy as np

def make_tuple():
    return (np.random.uniform(0,1), np.random.choice(['Eilif', 'Katie', 'Carl']))

ar = np.array([make_tuple() for i in range(1000)], dtype=[('height', 'f8'), ('name', 'a10')])
print(ar['name'])