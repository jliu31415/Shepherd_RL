import numpy as np

# distance from [a1, a2] to [b1, b2]
def dist(a, b=[0, 0]):
    return np.linalg.norm(np.subtract(a, b))

# unit vect from [b1, b2] to [a1, a2]
def unit_vect(a, b=[0, 0]):
    return np.subtract(a, b)/(dist(a, b) + 1e-5)    # prevent division by zero

# get random unit vector
def rand_unit():
    return unit_vect([np.random.rand()-.5, np.random.rand()-.5])