# torch imports
import torch
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        
        # Add a fully connected layer
        self.fc1 = nn.Linear(input_features, hidden_dim)
        
        # Add a fully connected layer  
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        
        # Add a fully connected layer
        self.fc3 = nn.Linear(int(hidden_dim/2), output_dim)
        
        # Add dropout to avoid overfitting
        self.drop = nn.Dropout(0.25)
        
        # add a  element-wise function sigmoid
        self.sigmoid = nn.Sigmoid()

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        
        # Add a fully connected layer with Relu activation
        x = torch.relu(self.fc1(x))
        
        # Add a dropout to avoid overfitting
        x = self.drop(x)
        
        # Add a fully connected layer with Relu activation function
        x = torch.relu(self.fc2(x))
        
        # Generate single, sigmoid-activated value as output
        x = torch.sigmoid(self.fc3(x))
        
        return x
    