import numpy as np

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = -1/m*(np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

Y = np.asarray([[1, 1, 0]])
AL = np.array([[.8,.9,0.4]])
print("cost = " + str(compute_cost(AL, Y)))
