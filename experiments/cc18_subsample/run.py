"""In this script the objective is to test the main training paradigm. 
    We first define an MLP model, take one of the CC-18 datasets, and 
    performed a 3-fold cross validation using MCDropout for uncertainty estimation.
    We hardcode some MCDropout hyperparameters, which are the ones later
    to be run over for the purpose of our experiments.
"""

import random
from sys import path
from typing import Any

import os
import numpy as np
import torch

# import dataset and dataloader for pytoarch
import torch.utils.data
from tqdm.auto import tqdm
import itertools
import argparse

path.append("./")

from src.utils.data import get_dataset, load_dataset_subsample
from src.utils.io import load_config
from src.utils import OutputTypeError
from src.train import train


# Add argparse for subset_id
parser = argparse.ArgumentParser()
parser.add_argument(
    "--subset_id", help="Identifier for the subset to focus on", default=0
)
parser.add_argument(
    "--error_handling", type=str, help="Error handling method", default="ignore"
)
parser.add_argument(
    "--config_name", type=str, help="Path to the config file", default="config.yaml"
)
args = parser.parse_args()

subset_id = args.subset_id
error_handling = args.error_handling

all_dataset_ranges = {
    0: range(0, 4),
    1: range(4, 8),
    2: range(8, 12),
    3: range(12, 16),
    4: range(16, 18),
    5: range(18, 20),
    'all': range(0, 20),
}
dataset_id_s = list(
    all_dataset_ranges[subset_id]
)  # this identifies the dataset inside the OpenML-CC18 benchmark suitea


def set_seed(random_seed: int) -> None:
    # set reproduction seeds
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # set numpy seed
    np.random.seed(random_seed)

    # set any other random seed
    random.seed(random_seed)


# NOTE: we decided to avoid this part, since it can make the training a lot slower
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def parallelizible_single_train(
    dataset_id: int,
    dropout_rate: float,
    model_args: dict,
    train_args: dict,
    results_path: str,
    datasets_to_use: list[int],
    random_seed: int,
) -> None:

    task_num = int(datasets_to_use[dataset_id])

    print(f"Training on dataset {task_num} from the OpenML-CC18 benchmark suite")
    x, y, name, task_type, output_size = get_dataset(task_num=task_num)
    print(f"Dataset: {name}")

    # Define the model
    input_size = x.shape[1]

    model_args.update(
        dict(input_size=input_size, output_type=task_type, output_size=output_size)
    )
    best_model_infos = train(
        x=x,
        y=y,
        task_num=task_num,
        num_folds=train_args["num_folds"],
        num_epochs=train_args["num_epochs"],
        learning_rate=train_args["learning_rate"],
        model_precision=train_args["model_precision"],
        batch_size=train_args["batch_size"],
        length_scale=train_args["length_scale"],
        learning_rate_epoch_rate=train_args["learning_rate_epoch_rate"],
        learning_rate_decay=train_args["learning_rate_decay"],
        random_seed=random_seed,
        dropout_rate=dropout_rate,
        model_args=model_args,
    )

    output_filename: str = (
        f"task{task_num}_dropout_rate{dropout_rate}_model_precision{train_args['model_precision']}_num_mcdropout_iterations{train_args['num_mcdropout_iterations']}_num_layers{train_args['num_layers']}.pth"
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


# Update the main function
def main():

    path_to_script: str = os.path.abspath(__file__)
    config_name: str = args.config_name
    path_to_config: str = os.path.join(os.path.dirname(path_to_script), config_name)

    configs: dict[str, Any] = load_config(path=path_to_config)
    hidden_activation_type = configs["hidden_activation_type"]
    batch_size = configs["batch_size"]
    length_scale = configs["length_scale"]
    starting_learning_rate = configs["starting_learning_rate"]
    learning_rate_decay = configs["learning_rate_decay"]
    learning_rate_epoch_rate = configs["learning_rate_epoch_rate"]
    num_epochs = configs["num_epochs"]
    num_crossval_folds = configs["num_crossval_folds"]
    prediction_threshold = configs["prediction_threshold"]
    random_seed = configs["random_seed"]
    layer_size = configs["layer_size"]
    results_path = configs["results_path"]
    subsample_path = configs["subsample_path"]
    dropout_rate_s = configs["dropout_rate_s"]
    model_precision_s = configs["model_precision_s"]
    num_mcdropout_iterations_s = configs["num_mcdropout_iterations_s"]
    num_layers_s = configs["num_layers_s"]

    set_seed(random_seed=random_seed)

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
                    results_path=results_path,
                    datasets_to_use=datasets_to_use,
                    model_args=dict(
                        num_layers=num_layers,
                        hidden_layer_size=layer_size,
                        hidden_activation_type=hidden_activation_type,
                        num_mcdropout_iterations=num_mcdropout_iterations,
                        dropout_rate=dropout_rate,
                        prediction_threshold=prediction_threshold,
                    ),
                    train_args=dict(
                        batch_size=batch_size,
                        length_scale=length_scale,
                        model_precision=model_precision,
                        learning_rate=starting_learning_rate,
                        learning_rate_decay=learning_rate_decay,
                        learning_rate_epoch_rate=learning_rate_epoch_rate,
                        num_epochs=num_epochs,
                        num_folds=num_crossval_folds,
                    ),
                    random_seed=random_seed,
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
            except OutputTypeError as e:
                print("CODE CRUSHED BECAUSE THERE IS A NEW PREDICTION TYPE")
                current_combination = dict(
                    dataset_id=dataset_id,
                    dropout_rate=dropout_rate,
                    model_precision=model_precision,
                    num_mcdropout_iterations=num_mcdropout_iterations,
                    num_layers=num_layers,
                )
                print(f"FAULTY COMBINATION: {current_combination}")
                raise e
        else:
            continue


if __name__ == "__main__":
    main()
