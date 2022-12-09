def foward_propogation(input_layer, weights):
    """
    Parameters:
    input_layer - ndarray of shape ((1 + INPUT_NODES) x 1)
    weights - a dictionary with ndarray of weights
    HIDDEN_NODES - an integer representing the number if hidden nodes in the second layer
    
    Returns:
    hidden_layer - ndarray of shape (1 + HIDDEN_NODES x 1)
    output_layer - ndarray of shape (INPUT_NODES x 1)
    
    Purpose:
    To compute a new hidden layer based off the weights of our model
    To compute a new output layer with the newly computed hidden layer
    """    
    # This function is vectorized using numpy
    # for incredibly fast computation!!!

    # applying our weights to the input layer via dot product and the tanh activation function + bias term
    hidden_layer = np.insert(tanh(np.dot(input_layer,weights['Layer 1'])), 0 , 1)

    # applying our last weights
    output_layer = np.dot(hidden_layer,weights['Layer 2'])

    return (hidden_layer, output_layer)
