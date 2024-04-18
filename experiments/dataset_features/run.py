"""In this script the objective is to test the main training paradigm. 
    We first define an MLP model, take one of the CC-18 datasets, and 
    performed a 3-fold cross validation using MCDropout for uncertainty estimation.
    We hardcode some MCDropout hyperparameters, which are the ones later
    to be run over for the purpose of our experiments.
"""

from sys import path
from typing import Any
from tqdm.auto import tqdm

import os

# import dataset and dataloader for pytoarch
import torch.utils.data
import argparse
from logging import getLogger, basicConfig, INFO

path.append("./")

from src.utils.data import load_dataset_subsample
from src.utils.io import load_config
from src.utils import set_seed
from src.data.dataset_measures import (
    dimensionality_stats,
    homogeneity_class_covariances,
    feature_correlation_class,
    normality_departure,
    information,
    class_stats,
)
from src.utils.data import get_dataset
from joblib import Parallel, delayed

logger = getLogger("run")

# Add argparse for subset_id
parser = argparse.ArgumentParser()
parser.add_argument(
    "--subset_id", help="Identifier for the subset to focus on", default="all"
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
    "all": range(0, 20),
}
dataset_id_s = list(
    all_dataset_ranges[subset_id]
)  # this identifies the dataset inside the OpenML-CC18 benchmark suitea


def extract_dataset_features(
    task_num: int, dataset_id: int, measures_to_use: list[callable], results_path: str, n_jobs: int = 1, random_seed: int = 0,
) -> dict:

    x, y, name, task_type, output_size = get_dataset(task_num=task_num)

    dataset_features = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(measure)(X_train=x, y_train=y, random_seed=random_seed) for measure in measures_to_use
    )
    
    # unravel the features, when they are tuples. Report features if they are float
    dataset_features = {
        feature_name: feature_val
        for features in dataset_features
        for feature_name, feature_val in features.items()
    }
    dataset_features.update(
        {
            "dataset_id": dataset_id,
            "task_name": name,
            "task_type": task_type,
            "output_size": output_size,
            "task_num": task_num,
        }
    )

    output_filename: str = f"datasetInfo_task{dataset_id}.pth"

    torch.save(dataset_features, os.path.join(results_path, output_filename))


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

    random_seed = configs["random_seed"]
    subsample_path = configs["subsample_path"]
    results_path = configs["results_path"]
    n_jobs = configs["num_jobs"]

    set_seed(random_seed=random_seed)


    datasets_to_use = load_dataset_subsample(subsample_path)

    measures_to_use = [
        dimensionality_stats,
        homogeneity_class_covariances,
        feature_correlation_class,
        normality_departure,
        information,
        class_stats,
    ]

    for dataset_id in tqdm(
        dataset_id_s, desc="Experiments", colour="magenta", total=len(dataset_id_s)
    ):
        try:
            extract_dataset_features(
                task_num=datasets_to_use[dataset_id],
                dataset_id=dataset_id,
                random_seed=random_seed,
                measures_to_use=measures_to_use,
                results_path=results_path,
                n_jobs=n_jobs,
            )
        except RuntimeError as e:
            print(f"CODE CRUSHED DUE TO THE FOLLOWING REASON: {e}")
            current_combination = dict(dataset_id=dataset_id)
            print(f"SKIPPING CURRENT COMBINATION: {current_combination}")
            if error_handling == "ignore":
                continue
            else:
                raise e


if __name__ == "__main__":
    main()
