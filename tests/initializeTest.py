import numpy as np
import sys

sys.path.insert(0, '../src/lowLevel/')

import initialize

def initialize_test():
    np.random.seed(3)

    parameters = initialize.initialize_parameters_deep([5,4,3])

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
