import numpy as np
def cartesian_product(x,y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])