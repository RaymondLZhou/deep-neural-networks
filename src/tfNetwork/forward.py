import tensorflow as tf

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1) 
    A1 = tf.nn.relu(Z1)                                      
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                  
    A2 = tf.nn.relu(Z2)                            
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3
