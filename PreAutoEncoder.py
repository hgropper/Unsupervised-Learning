import math # for sqrt, log, exponentials
import numpy as np # for vectorization and array
import random # for random simulation
import pandas as pd # for dataframe visualization
import matplotlib.pyplot as plt # for plotting data in a graph
import copy # for making predictions
from collections import OrderedDict # ordering dictionaries
import warnings # no annoying warnings

warnings.filterwarnings('ignore') # to ignore numpy's warnings

def generate_data_point(sigma):
    """
    Purpose:
    Generates a data point of at 30 dimensions.
    
    Parameters:
    sigma - a float number that alters our output, and adds more 
    noise (this should hinder the performance of our model)
    
    Returns:
    feature_vector - a list with a length of (dimensions + 1)
    where all elements are features
    """
    
    # intialize a feature vector of zeros
    feature_vector = np.zeros(30)
    
    # modifying x1
    feature_vector[0] = np.random.normal(0,1)
    
    # creating x4, x7, x10, x13, ... , x28
    indices_to_modify = np.array(list(range(4,28+3,3))) - 1
    for index in indices_to_modify:
        feature_vector[index] = feature_vector[index - 3] + np.random.normal(0,sigma**2)
    
    # modifying x2
    feature_vector[1] = feature_vector[0] + np.random.normal(0,sigma**2)
    
    # creating x5, x8, x11, ... , x29
    indices_to_modify = np.array(list(range(5,29+3,3))) - 1
    for index in indices_to_modify:
        feature_vector[index] = feature_vector[index - 3] + np.random.normal(0,sigma**2)
    
    # modifying x3
    feature_vector[2] = feature_vector[0] + np.random.normal(0,sigma**2)
    
    # creating x6, x9, x12, x15, ... , x30
    indices_to_modify = np.array(list(range(6,30+3,3))) - 1
    for index in indices_to_modify:
        feature_vector[index] = feature_vector[index - 3] + np.random.normal(0,sigma**2)
    
    return feature_vector

def generate_train_data_set(training_data_size = 5000, sigma = 0.10):
    """
    Purpose:
    To use the generate_data_point function to generate training
    data 
    
    Parameters:
    training_data_size - an integer specifying how many training data points
    you would like to generate
    
    sigma - a float number that alters our output
    
    Returns:
    x_train - ndarray with shape of (dimensions x number of data points)
    """
    
    # intialize our test and training data
    training_data = []
    
    # generating the training data
    for _ in range(0,training_data_size):
        training_data.append(generate_data_point(sigma))
        
    # putting our generated data into a numpy ndarray
    x_train = np.array(training_data)

    return x_train

# doing this so we do not have to calculate e
# everytime we run our activation function tanh
e = math.e

def tanh(z):
    """
    Purpose:
    Our activation function in our neural network
    
    Parameters:
    z - (30 x 1) vector containing random float values
    
    Returns:
    A value without bounds
    """
    pos_power = e ** z
    neg_power = e ** -z
    
    return (pos_power - neg_power) / (pos_power + neg_power)

def calculate_loss(x_train, x_predicted):
    """
    Purpose:
    calculates the loss between the train data points
    and the predicted data points
    
    Parameters:
    x_train - (5000 x 30) dimensional array
    x_predicted - (5000 x 30) dimensional array
    
    Returns:
    loss - a float value indicating our error
    """
    
    # number of data points
    N = len(x_train)
    
    # calculating the loss
    loss = (1 / N) * np.sum( (np.linalg.norm((x_train - x_predicted)))**2 )
    
    return loss
  
INPUT_NODES = 30 # input layer
HIDDEN_NODES = list(range(1,30+1)) # hidden layer
OUTPUT_NODES = 30 # output layer 

x_train = generate_train_data_set()
