import numpy as np


def Q_values(x, W1, W2, bias_W1, bias_W2):
    
    """
    Compute the Q values as ouput of the neural network.
    W1 and bias_W1 refer to the first layer
    W2 and bias_W2 refer to the second layer
    Use rectified linear units
    The output vectors of this function are Q and out1
    Q is the ouptut of the neural network: the Q values
    out1 contains the activation of the nodes of the first layer
    """


    # Neural activation: input layer -> hidden layer
    
    act1 = np.dot(W1,x) + bias_W1
    # Apply the sigmoid function
    out1 = 1 / (1 + np.exp(-act1))

    # Neural activation: hidden layer -> output layer
    
    act2 = np.dot(W2, out1) + bias_W2
    # Apply the sigmoid function
    Q = 1 / (1 + np.exp(-act2))
    return Q, out1