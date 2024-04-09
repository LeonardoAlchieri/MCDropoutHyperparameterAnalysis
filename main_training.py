"""In this script the objective is to test the main training paradigm. 
    We first define an MLP model, take one of the CC-18 datasets, and 
    performed a 3-fold cross validation using MCDropout for uncertainty estimation.
    We hardcode some MCDropout hyperparameters, which are the ones later
    to be run over for the purpose of our experiments.
"""

from math import isnan
import random

import os
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm.auto import tqdm
from gc import collect as pick_up_trash
import itertools
import argparse

# Add argparse for subset_id
parser = argparse.ArgumentParser()
parser.add_argument("--subset_id", type=int, help="Identifier for the subset to focus on", default=0)
parser.add_argument("--error_handling", type=str, help="Error handling method", default="ignore")
args = parser.parse_args()

subset_id = args.subset_id
error_handling = args.error_handling

all_dataset_ranges = {
    0: range(0,4),
    1: range(4,8),
    2: range(8,12),
    3: range(12,16),
    4: range(16,18),
    5: range(18,20),
}

# FIXED PARAMETERS
hidden_activation_type: str = "relu"
batch_size: int = 32
length_scale: float = (
    0.1  # defines the length scale for the L2 regularization term. Basically, how much you have to rescale the term.
)
starting_learning_rate: float = 0.01
learning_rate_decay: float = 0.5
learning_rate_epoch_rate: int = 5
num_epochs: int = 50
num_crossval_folds: int = 3
prediction_threshold: float = 0.5  # threshold for binary classification
random_seed: int = 42
layer_size: int = 1000
results_path: str = "./results.nosync/"
subsample_path: str = "./subsampled_tasks.csv"

# HYPERPARAMETERS TO INVESTIGATE
dataset_id_s: list[int] = list(
    all_dataset_ranges[subset_id]
)  # this identifies the dataset inside the OpenML-CC18 benchmark suitea
dropout_rate_s: float = [0.001, 0.05, 0.1, 0.5, 0.9]
model_precision_s: float = [
    0.001,
    0.05,
    0.5,
    0.9,
]  # also known as "tau". Defines the L2 regularization term
num_mcdropout_iterations_s: int = [3, 5, 10, 20, 100]
num_layers_s: int = [1, 3, 5, 10]


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


def get_dataset(task_num: int) -> tuple[torch.Tensor, torch.Tensor, str, str, int]:
    # 99 is the ID of the OpenML-CC18 study
    test_task = openml.tasks.get_task(task_num)
    test_dataset_obj = test_task.get_dataset()
    test_dataset = test_dataset_obj.get_data()

    x = test_dataset[0].drop(columns=[test_task.target_name])
    x = pd.get_dummies(x)
    # substitute the NaN values with the mean of the column
    x = x.fillna(x.mean())
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
        input_size: int,
        num_layers: int,
        layer_size: int,
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
        self.layers.append(nn.BatchNorm1d(1000))
        self.layers.append(self.activation)
        self.layers.append(nn.Dropout(dropout_rate))  # Dropout layer
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(1000, 1000))  # Hidden layers
            self.layers.append(nn.BatchNorm1d(1000))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(dropout_rate))  # Dropout layer

        self.output_layer = nn.Linear(layer_size, output_size)

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
        if torch.isnan(x).any():
            raise ValueError(f"Found NaN values in the input tensor {x}")
        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.output_layer(x)
        x = self.output_activation(x)
        return x


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def eval_mc_dropout(
    model: MLP,
    x: torch.Tensor,
    num_mcdropout_iterations: int,
    task_type: str,
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
    dropout_sample = [model(x) for _ in range(num_mcdropout_iterations)]
    mean = torch.nanmean(torch.stack(dropout_sample), dim=0)
    if task_type == "regression":
        uncertainty = nanvar(torch.stack(dropout_sample), dim=0)
    elif task_type == "classification" or task_type == "binary classification":
        uncertainty = -torch.nansum(mean * torch.log(mean), dim=1)
    else:
        raise ValueError(
            f"Task type {task_type} not supported. Supported types are: regression, classification. Found {task_type}."
        )

    return dropout_sample, mean, uncertainty


def train(
    x: torch.Tensor,
    y: torch.Tensor,
    num_folds: int,
    num_epochs: int,
    learning_rate: int,
    dropout_rate: float,
    model_precision: float,
    model_args: dict,
    task_num: int,
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
        task_type = model.task_type

        # Define the loss function based on the task type
        if task_type == "binary classification":
            loss_function = F.binary_cross_entropy
        elif task_type == "classification":
            loss_function = F.cross_entropy
        elif task_type == "regression":
            raise NotImplementedError(
                "Regression task type not implemented yet. Please implement the loss function for regression."
            )
        else:
            raise ValueError(
                f"Task type {task_type} not supported. Supported types are: regression, classification. Found {task_type}."
            )

        # Split the data into training and validation sets
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # # Convert the data to tensors
        x_train = x_train.float()
        y_train = y_train.float()
        x_val = x_val.float()
        y_val = y_val.float()

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
            0.0  # NOTE: might consider using the validation loss instead of accuracy — also acc not the best metric!
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
                train_dataloader,
                desc="Training Batches",
                colour="yellow",
                leave=False,
                disable=True,
            ):

                optimizer.zero_grad()

                # Forward pass
                y_pred = model(x_train)

                # Calculate the loss
                loss = loss_function(y_pred, y_train)

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()

                train_loss += loss.item()

            # Evaluate the model on the validation set
            # model.eval()
            with torch.no_grad():
                total_val_loss = 0
                val_uncertainties = []
                all_y_val_pred = torch.Tensor()
                all_y_val = torch.Tensor()
                # validate over the validation set
                for x_val, y_val in tqdm(
                    val_dataloader,
                    desc="Validation Batches",
                    colour="blue",
                    leave=False,
                    disable=True,
                ):
                    # Perform Monte Carlo Dropout during evaluation
                    (
                        y_val_pred_all_samples,
                        y_val_pred_mean,
                        y_val_pred_uncertainty,
                    ) = eval_mc_dropout(
                        model=model,
                        x=x_val,
                        num_mcdropout_iterations=model_args["num_mcdropout_iterations"],
                        task_type=task_type,
                    )
                    all_y_val_pred = torch.cat(
                        [all_y_val_pred, y_val_pred_mean.cpu()], dim=0
                    )
                    all_y_val = torch.cat([all_y_val, y_val.cpu()], dim=0)
                    
                    # Calculate the validation loss
                    val_loss = loss_function(y_val_pred_mean, y_val)
                    val_uncertainties.append(y_val_pred_uncertainty)
                    
                    total_val_loss += val_loss.item()
                    
                    
                if task_type == "binary classification" or task_type == "classification":
                    val_accuracy = accuracy_score(
                        y_val.cpu().numpy(),
                        (y_val_pred_mean > prediction_threshold).int().cpu().numpy(),
                    )
                    val_f1 = f1_score(
                        y_val.cpu().numpy(),
                        (y_val_pred_mean > prediction_threshold).int().cpu().numpy(),
                        average="binary" if task_type == "binary classification" else "macro",
                    )
                    # val_mcc = matthews_corrcoef(
                    #     y_val.cpu().numpy(),
                    #     (y_val_pred_mean > prediction_threshold).int().cpu().numpy(),
                    # )
                elif task_type == "regression":
                    raise NotImplementedError(
                        "Regression task type not implemented yet. Please implement the loss function for regression."
                    )
                else:
                    raise ValueError(
                        f"Task type {task_type} not supported. Supported types are: regression, classification. Found {task_type}."
                    )

                    

                # convert the list of uncertainties to a tensor. consider that the tensors might have different shape
                val_uncertainties = torch.cat(val_uncertainties)
                mean_val_uncertainty = torch.mean(val_uncertainties)
                val_accuracy = val_accuracy / len(val_dataset)
                # Print the training and validation loss for each epoch
                if val_accuracy > best_val_acc:
                    best_model_info = {
                        # "model_weights": model.state_dict(),
                        "training_info": {
                            "model": model.__class__.__name__,
                            "dataset_openml_id": task_num,
                            "dropout_rate": dropout_rate,
                            "model_precision": model_precision,
                            "num_mcdropout_iterations": model_args[
                                "num_mcdropout_iterations"
                            ],
                            "num_layers": model_args["num_layers"],
                        },
                        # "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": val_accuracy,
                        "val_f1": val_f1,
                        # "val_mcc": val_mcc,
                        "val_loss": val_loss.item(),
                        "mean_val_uncertainty": mean_val_uncertainty.item(),
                        "train_loss": train_loss,
                        "cross_val_fold": fold,
                        "random_seed": random_seed,
                    }
                    print(
                        f"""
                        task{task_num} 
                        dropout_rate{dropout_rate} 
                        model_precision{model_precision} 
                        num_mcdropout_iterations{model_args[
                                'num_mcdropout_iterations'
                            ]} 
                        num_layers{model_args['num_layers']}
                        current_learning_rate{param_group['lr']}
                        Epoch {epoch+1}/{num_epochs} — Training Loss: {loss.item()}
                        — Validation Loss: {val_loss.item()}
                        — Mean Validation Uncertainty: {mean_val_uncertainty.item()}
                        — Validation Accuracy: {val_accuracy}
                        - Validation F1: {val_f1}
                        """
                    )

        best_model_infos.append(best_model_info)
        del model
        pick_up_trash()

    return best_model_infos


def parallelizible_single_train(
    dataset_id: int,
    dropout_rate: float,
    model_precision: float,
    num_mcdropout_iterations: int,
    num_layers: int,
    layer_size: int,
    results_path: str,
    datasets_to_use: list[int],
) -> None:

    task_num = int(datasets_to_use[dataset_id])

    print(f"Training on dataset {task_num} from the OpenML-CC18 benchmark suite")
    x, y, name, task_type, output_size = get_dataset(task_num=task_num)
    print(f"Dataset: {name}")

    # Define the model
    input_size = x.shape[1]
    best_model_infos = train(
        x,
        y,
        num_folds=num_crossval_folds,
        num_epochs=num_epochs,
        learning_rate=starting_learning_rate,
        dropout_rate=dropout_rate,
        model_precision=model_precision,
        task_num=task_num,
        model_args=dict(
            input_size=input_size,
            num_layers=num_layers,
            layer_size=layer_size,
            hidden_activation_type=hidden_activation_type,
            output_type=task_type,
            output_size=output_size,
            num_mcdropout_iterations=num_mcdropout_iterations,
            dropout_rate=dropout_rate,
        ),
    )

    output_filename: str = (
        f"task{task_num}_dropout_rate{dropout_rate}_model_precision{model_precision}_num_mcdropout_iterations{num_mcdropout_iterations}_num_layers{num_layers}.pth"
    )
    # save list of dicts to json
    # TODO: find a better schema. Probably not a good idea to save everything at the end.
    torch.save(best_model_infos, os.path.join(results_path, output_filename))


def get_already_run_experiments(
    path_to_ressults_folder: str,
) -> list[tuple[int | float]]:
    result_files_raw: list[str] = os.listdir(path_to_ressults_folder)
    # NOTE: example output
    # task0_dropout_rate0.9_model_precision0.9_num_mcdropout_iterations5_num_layers5.pth
    return [
        (
            int(file.split("_")[0].replace("task", "")),
            float(file.split("_")[2].replace("rate", "")),
            float(file.split("_")[4].replace("precision", "")),
            int(file.split("_")[7].replace("iterations", "")),
            int(file.split("_")[9].replace("layers", "").replace(".pth", "")),
        )
        for file in result_files_raw
    ]


def load_dataset_subsample(file_path: str) -> list:
    with open(file_path, "r") as file:
        dataset_subsample = [(line.strip()) for line in file]

    return dataset_subsample


# Update the main function
def main():

    # TODO: move global variables to config file

    datasets_to_use = load_dataset_subsample(subsample_path)

    previous_experiments = get_already_run_experiments(results_path)

    for (
        dataset_id,
        dropout_rate,
        model_precision,
        num_mcdropout_iterations,
        num_layers,
    ) in tqdm(
        itertools.product(
            dataset_id_s,
            dropout_rate_s,
            model_precision_s,
            num_mcdropout_iterations_s,
            num_layers_s,
        ),
        desc="Experiments",
        colour="red",
    ):
        if (
            dataset_id,
            dropout_rate,
            model_precision,
            num_mcdropout_iterations,
            num_layers,
        ) not in previous_experiments:
            try:
                parallelizible_single_train(
                    dataset_id=dataset_id,
                    dropout_rate=dropout_rate,
                    model_precision=model_precision,
                    num_mcdropout_iterations=num_mcdropout_iterations,
                    num_layers=num_layers,
                    layer_size=layer_size,
                    results_path=results_path,
                    datasets_to_use=datasets_to_use,
                )
            except RuntimeError as e:
                print(f"CODE CRUSHED DUE TO THE FOLLOWING REASON: {e}")
                current_combination = dict(
                    dataset_id=dataset_id,
                    dropout_rate=dropout_rate,
                    model_precision=model_precision,
                    num_mcdropout_iterations=num_mcdropout_iterations,
                    num_layers=num_layers,
                )
                print(f"SKIPPING CURRENT COMBINATION: {current_combination}")
                if error_handling == "ignore":
                    continue
                else:
                    raise e
        else:
            continue


if __name__ == "__main__":
    main()
