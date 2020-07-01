import sys
import tensorflow as tf

sys.path.insert(0, '../../src/tfNetwork/')

import initialize

def initialize_test():
    X, Y = initialize.create_placeholders(12288, 6)
    print ("X = " + str(X))
    print ("Y = " + str(Y))

    tf.reset_default_graph()
    with tf.Session() as sess:
        parameters = initialize.initialize_parameters()
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))
