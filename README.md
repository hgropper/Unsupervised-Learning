# Homework-3-Machine-Learning-Unsupervised-Learning
We look at the structure of data and not its corresponding output value, just the data matrix. By looking at the structure of data we can better understand it. In this project, I am reducing a 30 dimensional data set into K dimensions, and then back into its original 30 dimensions. This is game-changing! If I am able to reduce a dataset into K dimensions and I am able to blow it back up to 30 dimensions, then I have successfully compressed the data. 

## How am I compressing the data?
I am coding a neural network (auto-encoder) from scratch. This neural network has 3 layers: input layer, hidden layer, and an output layer. These layers are fully connected and the weights of each neuron is updated using stochastic gradient descent. 

### Other ways to compress the data
In addition to an auto-encoder you can also use PCA (principal component analysis) to reduce the dimensions of a dataset.
