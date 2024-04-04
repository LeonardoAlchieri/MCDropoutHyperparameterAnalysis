"""In this script the objective is to test the main training paradigm. 
    We first define an MLP model, take one of the CC-18 datasets, and 
    performed a 3-fold cross validation using MCDropout for uncertainty estimation.
    We hardcode some MCDropout hyperparameters, which are the ones later
    to be run over for the purpose of our experiments.
"""

import openml
from numpy import ndarray
import torch
import torch.nn as nn

task_num: int = 0  # this identifies the dataset inside the OpenML-CC18 benchmark suite
hidden_activation_type: str = "relu"

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import torch.optim as optim
import torch.nn.functional as F

# Convert the class labels to one-hot encoded vectors

def prepare_prediction_array(y: ndarray) -> torch.Tensor:
    # Use LabelEncoder to encode the class array
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)

    num_classes = len(set(encoded_labels))

    # Convert the encoded labels to one-hot encoded vectors using PyTorch
    class_array_one_hot = torch.nn.functional.one_hot(torch.tensor(encoded_labels), num_classes)
    return class_array_one_hot
    

def get_dataset() -> tuple[ndarray, ndarray, str, str, int]:
    # 99 is the ID of the OpenML-CC18 study
    cc18_suite = openml.study.get_suite(99)
    tasks = cc18_suite.tasks
    test_task = openml.tasks.get_task(tasks[task_num])
    test_dataset_obj = test_task.get_dataset()
    test_dataset = test_dataset_obj.get_data()

    x = test_dataset[0].drop(columns=[test_task.target_name])
    y = test_dataset[0][test_task.target_name]
    name = test_dataset_obj.name
    y = prepare_prediction_array(y)
    prediction_type = test_task.task_type
    if prediction_type == 'Supervised Classification':
        prediction_type = 'multiclass classification' if len(y[0]) > 1 else 'binary classification'
        # TODO: I should implement the multilabel classification
    elif 'Regression' in prediction_type:
        prediction_type = 'regression'
    else:
        raise ValueError(f"Prediction type {prediction_type} not supported. Supported types are: regression, binary classification, multiclass classification, multilabel classification")
    
    output_size = len(y[0])
    return x, y, name, prediction_type, output_size


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers,
        hidden_activation_type: str = "relu",
        output_type: str = "regression",
        output_size: int = 1,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, 1000))  # First layer

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(1000, 1000))  # Hidden layers

        self.layers.append(nn.Linear(1000, 1))  # Output layer
        if hidden_activation_type == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Activation type {hidden_activation_type} not supported. Supported types are: relu.")
         
        if output_type == "regression":
            self.output_activation = nn.Identity()
        elif output_type == "binary classification":
            self.output_activation = nn.Sigmoid()
        elif output_type == "multiclass classification":
            self.output_activation = nn.Softmax(dim=1)
        elif output_type == "multilabel classification":
            self.output_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Output type {output_type} not supported. Supported types are: regression, binary classification, multiclass classification, multilabel classification")
            
        
        self.input_layer = nn.Linear(input_size, 1000)
        self.output_layer = nn.Linear(1000, output_size)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x

    

def train(model, x, y, num_folds, num_epochs, learning_rate, mc_dropout_prob):
    # Define the loss function based on the task type
    if model.output_activation == nn.Sigmoid():
        loss_function = F.binary_cross_entropy
    elif model.output_activation == nn.Softmax(dim=1):
        loss_function = F.cross_entropy
    else:
        raise ValueError("Invalid output activation function")
    
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Perform k-fold cross validation
    kf = KFold(n_splits=num_folds)
    fold = 1
    for train_index, val_index in kf.split(x):
        print(f"Training on fold {fold}")
        fold += 1
        
        # Split the data into training and validation sets
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Convert the data to tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        
        # Train the model for the specified number of epochs
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Perform Monte Carlo Dropout during training
            model.train_mc_dropout(mc_dropout_prob)
            
            # Forward pass
            y_pred = model(x_train)
            
            # Calculate the loss
            loss = loss_function(y_pred, y_train)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Evaluate the model on the validation set
            model.eval()
            with torch.no_grad():
                # Perform Monte Carlo Dropout during evaluation
                model.eval_mc_dropout(mc_dropout_prob)
                
                # Forward pass
                y_val_pred = model(x_val)
                
                # Calculate the validation loss
                val_loss = loss_function(y_val_pred, y_val)
                
                # Print the training and validation loss for each epoch
                print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {loss.item()} - Validation Loss: {val_loss.item()}")

# Update the main function
def main():
    x, y, name, task_type, output_size = get_dataset()

    # Define the model
    input_size = x.shape[1]
    num_layers = int(input("Enter the number of layers: "))
    
    model = MLP(input_size, 
                num_layers, 
                hidden_activation_type=hidden_activation_type, 
                output_type=task_type, 
                output_size=output_size)




if __name__ == "__main__":
    main()
