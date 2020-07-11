import numpy as np
import sys
import tensorflow as tf

sys.path.insert(0, '../../src/tfNetwork/')

import cost
import forward
import initialize

def cost_test():
    tf.reset_default_graph()

    with tf.Session() as sess:
        X, Y = initialize.create_placeholders(12288, 6)
        parameters = initialize.initialize_parameters()
        Z3 = forward.forward_propagation(X, parameters)
        cost = cost.compute_cost(Z3, Y)
        print("cost = " + str(cost))
