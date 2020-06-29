import numpy as np
import sys

sys.path.insert(0, '../src/lowLevel/')

import update

def update_test():
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)

    parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2}

    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)

    grads = {"dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}

    parameters = update.update_parameters(parameters, grads, 0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))
