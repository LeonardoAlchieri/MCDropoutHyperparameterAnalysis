"""In this script the objective is to test the main training paradigm. 
    We first define an MLP model, take one of the CC-18 datasets, and 
    performed a 3-fold cross validation using MCDropout for uncertainty estimation.
    We hardcode some MCDropout hyperparameters, which are the ones later
    to be run over for the purpose of our experiments.
"""

from sys import path
from typing import Any

import os

# import dataset and dataloader for pytoarch
from tqdm.auto import tqdm
import itertools
import argparse
from warnings import warn
from joblib import Parallel, delayed
from logging import getLogger, basicConfig, INFO

path.append("./")

from src.utils.data import load_dataset_subsample, get_outer_fold
from src.utils.io import load_config
from src.utils import OutputTypeError
from src.train import train
from src.utils import set_seed

logger = getLogger("run")

# Add argparse for subset_id
parser = argparse.ArgumentParser()
parser.add_argument(
    "--subset_id", help="Identifier for the subset to focus on", default="test"
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
    "0": range(0, 4),
    "1": range(4, 8),
    "2": range(8, 12),
    "3": range(12, 16),
    "4": range(16, 18),
    "5": range(18, 20),
    "all": range(0, 20),
    "test": [10],
}
dataset_id_s = list(
    all_dataset_ranges[subset_id]
)  # this identifies the dataset inside the OpenML-CC18 benchmark suitea


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


def train_parallel(
    task_num,
    dataset_id,
    num_inner_folds,
    results_path,
    random_seed,
    outer_fold_idxs,
    outer_fold_id,
    experiment_args,
    model_args,
    train_args,
    num_jobs,
    outlier_flag,
    outer_fold_idxs_s,
):
    for outer_fold_id, outer_fold_idxs in outer_fold_idxs_s[task_num].items():
        train(
            task_num=task_num,
            dataset_id=dataset_id,
            num_inner_folds=num_inner_folds,
            results_path=results_path,
            random_seed=random_seed,
            outer_fold_idxs=outer_fold_idxs,
            outer_fold_id=outer_fold_id,
            experiment_args=experiment_args,
            model_args=model_args,
            train_args=train_args,
            num_jobs=num_jobs,
            outlier_flag=outlier_flag,
        )


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
    hidden_activation_type = configs["hidden_activation_type"]
    # batch_size = configs["batch_size"]
    # length_scale = configs["length_scale"]
    # starting_learning_rate = configs["starting_learning_rate"]
    # learning_rate_decay = configs["learning_rate_decay"]
    # learning_rate_epoch_rate = configs["learning_rate_epoch_rate"]
    num_epochs = configs["num_epochs"]
    num_crossval_folds = configs["num_crossval_folds"]
    # prediction_threshold = configs["prediction_threshold"]
    random_seed = configs["random_seed"]
    layer_size = configs["layer_size"]
    results_path = configs["results_path"]
    subsample_path = configs["subsample_path"]
    dropout_rate_s = configs["dropout_rate_s"]
    model_precision_s = configs["model_precision_s"]
    num_mcdropout_iterations_s = configs["num_mcdropout_iterations_s"]
    num_layers_s = configs["num_layers_s"]
    num_jobs = configs["num_jobs"]
    path_to_fold_info = configs["path_to_fold_info"]
    outlier_only_flag = configs.get("outlier_only_flag", False)

    set_seed(random_seed=random_seed)

    # TODO: move global variables to config file

    datasets_to_use = load_dataset_subsample(subsample_path)

    previous_experiments = get_already_run_experiments(results_path)
    logger.info(f"{previous_experiments=}")

    # the keys of outer_fold_idxs_s are the ids of the outer fold, used to check if
    # there is a previous experiment for that fold
    outer_fold_idxs_s = get_outer_fold(path=path_to_fold_info, fold_type="train")

    if num_jobs == 1:
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
            colour="magenta",
            total=len(dataset_id_s)
            * len(dropout_rate_s)
            * len(model_precision_s)
            * len(num_mcdropout_iterations_s)
            * len(num_layers_s),
        ):
            if (
                dataset_id,
                dropout_rate,
                model_precision,
                num_mcdropout_iterations,
                num_layers,
            ) not in previous_experiments:
                logger.info(
                    f"Running experiment with combination {(dataset_id,dropout_rate,model_precision,num_mcdropout_iterations,num_layers)}."
                )
                for outer_fold_id, outer_fold_idxs in outer_fold_idxs_s[
                    datasets_to_use[dataset_id]
                ].items():
                    try:
                        train(
                            task_num=datasets_to_use[dataset_id],
                            dataset_id=dataset_id,
                            num_inner_folds=num_crossval_folds,
                            results_path=results_path,
                            random_seed=random_seed,
                            outer_fold_idxs=outer_fold_idxs,
                            outer_fold_id=outer_fold_id,
                            experiment_args={
                                "dropout_rate": dropout_rate,
                                "alpha": model_precision,
                                "mcdropout_num": num_mcdropout_iterations,
                                "num_layers": num_layers,
                            },
                            model_args={
                                "layer_size": layer_size,
                                "hidden_activation_type": hidden_activation_type,
                            },
                            train_args={"num_epochs": num_epochs},
                            num_jobs=num_jobs,
                            outlier_flag=outlier_only_flag,
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
    else:
        warn('Error handling is set to "raise" for parallel processing.')
        Parallel(n_jobs=num_jobs, backend="loky")(
            delayed(train_parallel)(
                task_num=datasets_to_use[dataset_id],
                dataset_id=dataset_id,
                num_inner_folds=num_crossval_folds,
                results_path=results_path,
                random_seed=random_seed,
                outer_fold_idxs=...,
                outer_fold_id=...,
                experiment_args={
                    "dropout_rate": dropout_rate,
                    "alpha": model_precision,
                    "mcdropout_num": num_mcdropout_iterations,
                    "num_layers": num_layers,
                },
                model_args={
                    "layer_size": layer_size,
                    "hidden_activation_type": hidden_activation_type,
                },
                train_args={"num_epochs": num_epochs},
                num_jobs=num_jobs,
                outlier_flag=outlier_only_flag,
                outer_fold_idxs_s=outer_fold_idxs_s,
                
            )
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
                colour="magenta",
                total=len(dataset_id_s)
                * len(dropout_rate_s)
                * len(model_precision_s)
                * len(num_mcdropout_iterations_s)
                * len(num_layers_s),
            )
        )


if __name__ == "__main__":
    main()
