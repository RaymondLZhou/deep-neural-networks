import numpy as np
import forward

def predict(X, y, parameters, dataset):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1,m), dtype = np.int)
    
    probas, caches = forward.L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print(dataset + " accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
        
    return p
