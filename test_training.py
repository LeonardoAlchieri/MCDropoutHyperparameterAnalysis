"""In this script the objective is to test the main training paradigm. 
    We first define an MLP model, take one of the CC-18 datasets, and 
    performed a 3-fold cross validation using MCDropout for uncertainty estimation.
    We hardcode some MCDropout hyperparameters, which are the ones later
    to be run over for the purpose of our experiments.
"""

import json
import random

import numpy as np
import openml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import dataset and dataloader for pytoarch
import torch.utils.data
from accelerate import Accelerator
from numpy import ndarray
from pyparsing import col
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from gc import collect as pick_up_trash

# FIXED PARAMETERS
hidden_activation_type: str = "relu"
batch_size: int = 32
length_scale: float = (
    0.1  # defines the length scale for the L2 regularization term. Basically, how much you have to rescale the term.
)
starting_learning_rate: float = 0.0001
learning_rate_decay: float = 1
learning_rate_epoch_rate: int = 2
num_epochs: int = 50
num_crossval_folds: int = 3
prediction_threshold: float = 0.5  # threshold for binary classification
random_seed: int = 42

# HYPERPARAMETERS TO INVESTIGATE
task_num: int = 0  # this identifies the dataset inside the OpenML-CC18 benchmark suitea
dropout_rate: float = 0.3
model_precision: float = 0.9  # also known as "tau". Defines the L2 regularization term
num_mcdropout_iterations: int = 10
num_layers: int = 3


# set reproduction seeds
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# set numpy seed
np.random.seed(random_seed)

# set any other random seed
random.seed(random_seed)


# Convert the class labels to one-hot encoded vectors


def prepare_prediction_array(y: ndarray) -> torch.Tensor:
    # Use LabelEncoder to encode the class array
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)

    num_classes = len(set(encoded_labels))

    # Convert the encoded labels to one-hot encoded vectors using PyTorch
    class_array_one_hot = torch.nn.functional.one_hot(
        torch.tensor(encoded_labels), num_classes
    )
    if class_array_one_hot.shape[1] == 2:
        print(f"Binary classification detected. Converting to single vector")
        # Convert binary classification to a single vector
        class_array_one_hot = class_array_one_hot[:, 1].unsqueeze(1)
    return class_array_one_hot


def get_dataset() -> tuple[torch.Tensor, torch.Tensor, str, str, int]:
    # 99 is the ID of the OpenML-CC18 study
    cc18_suite = openml.study.get_suite(99)
    tasks = cc18_suite.tasks
    test_task = openml.tasks.get_task(tasks[task_num])
    test_dataset_obj = test_task.get_dataset()
    test_dataset = test_dataset_obj.get_data()

    x = test_dataset[0].drop(columns=[test_task.target_name])
    x = pd.get_dummies(x)
    x = torch.tensor(x.values.astype(float), dtype=torch.float32)

    y = test_dataset[0][test_task.target_name].to_numpy()
    name = test_dataset_obj.name
    y = prepare_prediction_array(y)

    prediction_type = test_task.task_type
    if prediction_type == "Supervised Classification":
        prediction_type = (
            "multiclass classification" if len(y[0]) > 1 else "binary classification"
        )
        # TODO: I should implement the multilabel classification
    elif "Regression" in prediction_type:
        prediction_type = "regression"
    else:
        raise ValueError(
            f"Prediction type {prediction_type} not supported. Supported types are: regression, binary classification, multiclass classification, multilabel classification"
        )

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
        num_mcdropout_iterations: int = 2,
        dropout_rate: float = 0.05,
    ) -> None:
        super(MLP, self).__init__()

        if hidden_activation_type == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Activation type {hidden_activation_type} not supported. Supported types are: relu."
            )

        # Define the main layers in the network. We use a simple structure
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, 1000))  # First layer
        self.layers.append(self.activation)
        self.layers.append(nn.Dropout(dropout_rate))  # Dropout layer
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(1000, 1000))  # Hidden layers
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(dropout_rate))  # Dropout layer

        if output_type == "regression":
            self.output_activation = nn.Identity()
            self.task_type = "regression"
        elif output_type == "binary classification":
            self.output_activation = nn.Sigmoid()
            self.task_type = "binary classification"
        elif output_type == "multiclass classification":
            self.output_activation = nn.Softmax(dim=1)
            self.task_type = "classification"
        elif output_type == "multilabel classification":
            self.output_activation = nn.Sigmoid()
            self.task_type = "classification"
        else:
            raise ValueError(
                f"Output type {output_type} not supported. Supported types are: regression, binary classification, multiclass classification, multilabel classification"
            )

        self.output_layer = nn.Linear(1000, output_size)

        if num_mcdropout_iterations > 1:
            self.num_mcdropout_iterations = num_mcdropout_iterations
        else:
            raise (
                ValueError(
                    f"num_mcdropout_iterations must be greater than 1. Found {num_mcdropout_iterations}."
                )
            )

        if dropout_rate > 0 and dropout_rate < 1:
            self.dropout_rate = dropout_rate
        else:
            raise (
                ValueError(
                    f"dropout_rate must be between 0 and 1. Found {dropout_rate}."
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x

    def eval_mc_dropout(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """This method runs the MCDropout at validation time. It returns a list of predictions

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]
            the output is composed as follow: a list of tensors, containing each predictions
            for the num_mcdropout_iterations performed to obtain a Montecarlo sample; the mean
            of the sample and the computed uncertainty (variance if regression, entropy if classification)
        """
        dropout_sample = [self.forward(x) for _ in range(self.num_mcdropout_iterations)]
        mean = torch.mean(torch.stack(dropout_sample), dim=0)
        if self.task_type == "regression":
            uncertainty = torch.var(torch.stack(dropout_sample), dim=0)
        elif (
            self.task_type == "classification"
            or self.task_type == "binary classification"
        ):
            uncertainty = -torch.sum(mean * torch.log(mean), dim=1)
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported. Supported types are: regression, classification. Found {self.task_type}."
            )

        return dropout_sample, mean, uncertainty


def train(
    x: torch.Tensor,
    y: torch.Tensor,
    num_folds: int,
    num_epochs: int,
    learning_rate: int,
    model_args: dict,
) -> list[dict]:

    print("Training the model...")
    accelerator = Accelerator()

    # Perform k-fold cross validation
    kf = KFold(n_splits=num_folds)
    fold = 1
    best_model_infos: list[dict] = []
    for train_index, val_index in tqdm(kf.split(x), desc="Folds", colour="magenta"):
        fold += 1

        # FIXME: I don't like that, in each fold, I am ridefining the model and the loss function. This might accidentaly break
        model = MLP(**model_args)

        # Define the loss function based on the task type
        if model.task_type == "binary classification":
            loss_function = F.binary_cross_entropy
        elif model.task_type == "classification":
            loss_function = F.cross_entropy
        elif model.task_type == "regression":
            raise NotImplementedError(
                "Regression task type not implemented yet. Please implement the loss function for regression."
            )
        else:
            raise ValueError(
                f"Task type {model.task_type} not supported. Supported types are: regression, classification. Found {model.task_type}."
            )

        # Split the data into training and validation sets
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Convert the data to tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        # Define the train and validation dataloaders
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )

        num_samples = len(x_train)
        reg = (
            length_scale**2 * (1 - dropout_rate) / (2.0 * num_samples * model_precision)
        )
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )

        # NOTE: I am defining their type  here, since accelerator messes them up
        model: MLP
        optimizer: optim.Adam

        # Since we are using MCDropout, we want the model to always be in training mode
        model.train()

        best_val_acc: float = (
            1.0  # NOTE: might consider using the validation loss instead of accuracy — also acc not the best metric!
        )
        best_model_info = dict()
        # Train the model for the specified number of epochs
        for epoch in tqdm(range(num_epochs), desc="Epochs", colour="green"):

            train_loss = 0
            if epoch % learning_rate_epoch_rate == 0:
                # lower the learning rate every 3 epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= learning_rate_decay

            for x_train, y_train in tqdm(
                train_dataloader, desc="Training Batches", colour="yellow", leave=False
            ):

                optimizer.zero_grad()

                # Forward pass
                y_pred = model(x_train)

                # Calculate the loss
                loss = loss_function(y_pred, y_train)

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()

                train_loss += loss.data.item()

            # Evaluate the model on the validation set
            # model.eval()
            with torch.no_grad():
                total_val_loss = 0
                val_accuracy = 0
                val_uncertainties = []
                # validate over the validation set
                for x_val, y_val in tqdm(
                    val_dataloader,
                    desc="Validation Batches",
                    colour="blue",
                    leave=False,
                ):
                    # Perform Monte Carlo Dropout during evaluation
                    (
                        y_val_pred_all_samples,
                        y_val_pred_mean,
                        y_val_pred_uncertainty,
                    ) = model.eval_mc_dropout(x_val)

                    # Calculate the validation loss
                    val_loss = loss_function(y_val_pred_mean, y_val)
                    val_uncertainties.append(y_val_pred_uncertainty)
                    if model.task_type == "binary classification":
                        val_accuracy += torch.sum(
                            (y_val_pred_mean > prediction_threshold).int() == y_val
                        )
                    elif model.task_type == "classification":
                        val_accuracy += torch.sum(
                            torch.argmax(y_val_pred_mean, dim=1)
                            == torch.argmax(y_val, dim=1)
                        )
                    elif model.task_type == "regression":
                        raise NotImplementedError(
                            "Regression task type not implemented yet. Please implement the loss function for regression."
                        )
                    else:
                        raise ValueError(
                            f"Task type {model.task_type} not supported. Supported types are: regression, classification. Found {model.task_type}."
                        )

                    total_val_loss += val_loss.data.item()

                # convert the list of uncertainties to a tensor. consider that the tensors might have different shape
                val_uncertainties = torch.cat(val_uncertainties)
                mean_val_uncertainty = torch.mean(val_uncertainties)
                val_accuracy = val_accuracy / len(val_dataset)
                # Print the training and validation loss for each epoch
                print(
                    f"Epoch {epoch+1}/{num_epochs} — Training Loss: {loss.item()}\
                        — Validation Loss: {val_loss.item()}\
                            — Mean Validation Uncertainty: {mean_val_uncertainty}\
                                — Validation Accuracy: {val_accuracy}"
                )
                if val_accuracy < best_val_acc:
                    best_model_info = {
                        "model_weights": model.state_dict(),
                        "training_info": {
                            "model": model.__class_.__name__,
                            "dataset_openml_id": task_num,
                            "dropout_rate": dropout_rate,
                            "model_precision": model_precision,
                            "num_mcdropout_iterations": num_mcdropout_iterations,
                            "num_layers": num_layers,
                        },
                        # "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": val_accuracy,
                        "val_loss": val_loss.item(),
                        "mean_val_uncertainty": mean_val_uncertainty.item(),
                        "train_loss": train_loss,
                        "cross_val_fold": fold,
                    }

        best_model_infos.append(best_model_info)
        del model
        pick_up_trash()

    return best_model_infos


# Update the main function
def main():
    print(f"Training on dataset {task_num} from the OpenML-CC18 benchmark suite")
    x, y, name, task_type, output_size = get_dataset()
    print(f"Dataset: {name}")

    # Define the model
    input_size = x.shape[1]

    best_model_infos = train(
        x,
        y,
        num_folds=num_crossval_folds,
        num_epochs=num_epochs,
        learning_rate=starting_learning_rate,
        model_args=dict(
            input_size=input_size,
            num_layers=num_layers,
            hidden_activation_type=hidden_activation_type,
            output_type=task_type,
            output_size=output_size,
            num_mcdropout_iterations=num_mcdropout_iterations,
            dropout_rate=dropout_rate,
        ),
    )

    # save list of dicts to json
    # TODO: find a better schema. Probably not a good idea to save everything at the end.
    torch.save(best_model_infos, "results/test_trainin.pth")


if __name__ == "__main__":
    main()
