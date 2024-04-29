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

# import dataset and dataloader for pytoarch
import torch.utils.data
from tqdm.auto import tqdm
import itertools
import argparse
from logging import getLogger, basicConfig, INFO

path.append("./")

from src.utils.data import load_dataset_subsample
from src.utils.io import load_config
from src.utils import OutputTypeError
from src.train.resnet import train
from src.utils import set_seed

logger = getLogger("run")

# Add argparse for subset_id
parser = argparse.ArgumentParser()
parser.add_argument(
    "--error_handling", type=str, help="Error handling method", default="ignore"
)
parser.add_argument(
    "--config_name", type=str, help="Path to the config file", default="config.yaml"
)
args = parser.parse_args()

error_handling = args.error_handling

def get_already_run_experiments(
    path_to_ressults_folder: str,
) -> list[tuple[int | float]]:
    result_files_raw: list[str] = os.listdir(path_to_ressults_folder)
    # NOTE: example output
    # task0_dropout_rate0.9_model_precision0.9_num_mcdropout_iterations5_num_layers5.pth
    return [
        (
            float(file.split("_")[2].replace("rate", "")),
        )
        for file in result_files_raw
    ]


# Update the main function
def main():

    path_to_script_folder: str = os.path.dirname(os.path.abspath(__file__))
    config_name: str = args.config_name

    basicConfig(
        filename=os.path.join(
            path_to_script_folder, f"{config_name.split('.')[0]}.log"
        ),
        level=INFO,
    )
    path_to_config: str = os.path.join(path_to_script_folder, config_name)

    configs: dict[str, Any] = load_config(path=path_to_config)

    num_epochs = configs["num_epochs"]
    random_seed = configs["random_seed"]
    results_path = configs["results_path"]
    dropout_rate_s = configs["dropout_rate_s"]
    num_jobs = configs["num_jobs"]
    
    outlier_only_flag = configs.get("outlier_only_flag", False)

    set_seed(random_seed=random_seed)

    previous_experiments = get_already_run_experiments(results_path)
    logger.info(f"{previous_experiments=}")

    for (
        dropout_rate,
    ) in tqdm(
        itertools.product(
            dropout_rate_s,
        ),
        desc="Experiments",
        colour="magenta",
        total=len(dropout_rate_s)
    ):
        if (
            dropout_rate,
        ) not in previous_experiments:
            logger.info(
                f"Running experiment with combination {(dropout_rate)}."
            )
            try:
                train(
                    results_path=results_path,
                    random_seed=random_seed,
                    experiment_args={
                        "dropout_rate": dropout_rate,
                    },
                    train_args={"num_epochs": num_epochs},
                    num_jobs=num_jobs,
                    outlier_flag=outlier_only_flag,
                )

            except RuntimeError as e:
                print(f"CODE CRUSHED DUE TO THE FOLLOWING REASON: {e}")
                current_combination = dict(
                    dropout_rate=dropout_rate,
                )
                print(f"SKIPPING CURRENT COMBINATION: {current_combination}")
                if error_handling == "ignore":
                    continue
                else:
                    raise e
            except OutputTypeError as e:
                print("CODE CRUSHED BECAUSE THERE IS A NEW PREDICTION TYPE")
                current_combination = dict(
                    dropout_rate=dropout_rate,
                )
                print(f"FAULTY COMBINATION: {current_combination}")
                raise e
        else:
            continue



if __name__ == "__main__":
    main()
