import sys
import tensorflow as tf

sys.path.insert(0, '../../src/tfNetwork/')

import forward
import initialize

def forward_test():
    tf.reset_default_graph()

    with tf.Session() as sess:
        X, Y = initialize.create_placeholders(12288, 6)
        parameters = initialize.initialize_parameters()
        Z3 = forward.forward_propagation(X, parameters)
        print("Z3 = " + str(Z3))
