import numpy as np
import sys

sys.path.insert(0, '../src/lowLevel/')

import cost

def cost_test():
    Y = np.asarray([[1, 1, 0]])
    AL = np.array([[.8,.9,0.4]])
    print("cost = " + str(cost.compute_cost(AL, Y)))
