import numpy as np
import matplotlib.pyplot as plt

import load
import predict
import initialize
import forward
import cost
import backward
import update

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3001, print_cost = False):
    lambd = 1
    costs = []
    parameters = initialize.initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = forward.L_model_forward(X, parameters)

        costVal = cost.compute_cost(AL, Y, parameters, lambd)

        grads = backward.L_model_backward(AL, Y, caches, lambd)

        parameters = update.update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, costVal))
        if print_cost and i % 100 == 0:
            costs.append(costVal)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

train_x_orig, train_y, test_x_orig, test_y, classes = load.load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255
test_x = test_x_flatten/255

layers_dims = [12288, 64, 16, 16, 1]

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=1501, print_cost=True)

pred_train = predict.predict(train_x, train_y, parameters, "Training")
pred_test = predict.predict(test_x, test_y, parameters, "Testing")
