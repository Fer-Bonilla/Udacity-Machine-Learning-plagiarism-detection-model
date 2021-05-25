Plagiarism detection model with Pytorch

The third project from Udacity's Machine Learning Engineer Nanodegree

Project sections:

- Problem understanding
- Pytorch Model Architecture
- Implementation
- Project structure
- Model performance
- Using the model

## Problem understanding

Build a plagiarism detector that examines a text file and performs binary classification; labeling that file as either plagiarized or not, depending on how similar the text file is to a provided source text.


## Pytorch Model Architecture

![Pytorch model](https://github.com/Fer-Bonilla/Udacity-Machine-Learning-plagiarism-detection-model/blob/main/notebook_ims/network_architecture.png)

## Project structure

The project structure is based on the Udacity's project template:

  ```
  + data                        + ¨g*.*                   Files with the text data 
                                +  file_information.csv   File with filename, task and category fields 

  + 1_Data_Exploration.ipynb                              Notebook with Data exploration

  + 2_Plagiarism_Feature_Engineering.ipynb                Notebook for data loading and transformation

  + 3_Training_a_Model.ipynb                              Notebook for training and deployment scripts

  + source_pytorch              + model.py                Pytorch Binary classification implementation
                                + predict.py              Python prediction implementation
                                + train.py                Train function implementation

  + plagiarism_data             + train.csv               Features selected for training
                                + test.py                 Features selected for test                                                       

  + notebook_ims                + *.png                   Images used in Notebooks

  + helpers.py                                            Helpers functions  required for execution

  + problem_unittests.py                                  Testing scripts
  ```

## Implementation

**Pytorch BinaryClassifier Model**

  ```Python
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
  ```

**Pytorch training function**

  ```Python
 import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

# imports the model in model.py by name
from model import BinaryClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from the train.csv file
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


# Provided training function
def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    # training loop is provided
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        total_loss = 0

        for batch in train_loader:
            # get data
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x)
            
            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        
        if epoch%10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    ## TODO: Add args for the three model parameters: input_features, hidden_dim, output_dim
    # Model Parameters
    parser.add_argument('--input_features', type=int, default=4,
                        help='input_features (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=20,
                        help='hidden_dim (default: 20)')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='output_dim (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)


    ## --- Your code here --- ##
    
    ## TODO:  Build the model by passing in the input params
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = BinaryClassifier(args.input_features, args.hidden_dim, args.output_dim).to(device)

    ## TODO: Define an optimizer and loss function for training
    optimizer = torch.optim.SGD(model.parameters(),args.lr)
    criterion = torch.nn.BCELoss()

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, args.epochs, criterion, optimizer, device)

    ## TODO: complete in the model_info by adding three argument names, the first is given
    # Keep the keys of this dictionary as they are 
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'epochs': args.epochs,
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
            'lr': args.lr,
        }
        torch.save(model_info, f)
        
    ## --- End of your code  --- ##
    

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

  ```

## Model performance

The project structure is based on the Udacity's project template:

Training Loss 
  	```
	Epoch: 10, Loss: 0.6487539325441632
	Epoch: 20, Loss: 0.613213564668383
	Epoch: 30, Loss: 0.5324599317141941
	Epoch: 40, Loss: 0.4539638161659241
	Epoch: 50, Loss: 0.3536339061600821
	Epoch: 60, Loss: 0.301504128745624
	Epoch: 70, Loss: 0.28896574676036835
	Epoch: 80, Loss: 0.2756422853895596
	Epoch: 90, Loss: 0.2657772238765444
	Epoch: 100, Loss: 0.22072330649409974
  	```

Accuracy
  ```
  1.0
  ```

## Using the model

  1. Execute the 2_Plagiarism_Feature_Engineering.ipynb (Load the data and write into plagirims_data directory to create train and test data)
  2. Execute 3_Training_a_Model.ipynb script to train and deploy the model
  3. Delete all the resources from Sagemaker

## Author 
Fernando Bonilla [linkedin](https://www.linkedin.com/in/fer-bonilla/)
