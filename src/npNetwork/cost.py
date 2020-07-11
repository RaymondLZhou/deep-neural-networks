import numpy as np

def compute_cost(AL, Y, parameters, lambd):
    m = Y.shape[1]

    entropyCost = -1/m*(np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))
    
    entropyCost = np.squeeze(entropyCost)
    assert(entropyCost.shape == ())

    regularizationCost = 0

    for l in range(1, len(parameters)//2):
        regularizationCost = regularizationCost + np.sum(np.square(parameters["W" + str(l)]))
    
    regularizationCost = regularizationCost*lambd/(2*m)
    cost = entropyCost + regularizationCost

    return cost
