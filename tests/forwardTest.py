import numpy as np
import sys

sys.path.insert(0, '../src/lowLevel/')

import forward

def forward_test():
    np.random.seed(1)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)

    Z, linear_cache = forward.linear_forward(A, W, b)
    print("Z = " + str(Z))

    np.random.seed(2)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)

    A, linear_activation_cache = forward.linear_activation_forward(A_prev, W, b, activation = "sigmoid")
    print("With sigmoid: A = " + str(A))

    A, linear_activation_cache = forward.linear_activation_forward(A_prev, W, b, activation = "relu")
    print("With ReLU: A = " + str(A))

    np.random.seed(6)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2,
                "W3": W3,
                "b3": b3}

    AL, caches = forward.L_model_forward(X, parameters)
    print("AL = " + str(AL))
    print("Length of caches list = " + str(len(caches)))
