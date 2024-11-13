import argparse
import os
from glob import glob
from logging import INFO, basicConfig, getLogger
from sys import path
from typing import Any
from warnings import warn

import pandas as pd
import torch
from tqdm.auto import tqdm
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import _is_fitted

path.append("./")

from src.utils.io import load_config, load_dataset_measures, load_prepare_uncertainties

logger = getLogger("run")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_name", type=str, help="Path to the config file", default="config.yaml"
)
args = parser.parse_args()


def prepare_symbolic_regression_model(
    outer_fold: int,
    n_iterations: int,
    n_population: int,
    path_to_save: str,
    random_state: int = 42,
    num_jobs: int = -1,
) -> PySRRegressor:

    save_directory: str = f"./temp_equation_files.nosync/outer_fold_{outer_fold}/"
    if not os.path.exists(save_directory):
        warn(f"Creating directory {save_directory}, since it did not exist.")
        os.makedirs(save_directory)

    return PySRRegressor(
        niterations=n_iterations,  # < Increase me for better results
        populations=n_population,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=[
            # "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
            "square",
            "cube",
            "exp",
            "abs",
            "log",
            "log10",
            "log2",
            "log1p",
            "sqrt",
        ],
        random_state=random_state,
        procs=num_jobs,
        multithreading=True,
        batching=True,
        temp_equation_file=True,
        tempdir=save_directory,
        delete_tempfiles=False,
        turbo=True,
        # equation_file=os.path.join(path_to_save, f"equation_{outer_fold}.csv"),
    )


def get_outer_fold_num(uncertainties: dict[str, pd.DataFrame]) -> int:
    outer_fold_max = [
        split_data["outer_fold"].unique().max() for split_data in uncertainties.values()
    ]
    return min(outer_fold_max) + 1


def train_test_symbolic_regression(
    model: PySRRegressor, uncertainties: dict[str, pd.DataFrame], outer_fold: int
) -> dict[str, Any]:

    x_train = uncertainties["validation"][
        uncertainties["validation"]["outer_fold"] == outer_fold
    ].drop(
        columns=[
            "entropies",
            "mutual_informations",
            "task_name",
            "outer_fold",
            "inner_fold",
            "task_num",
            "Unnamed: 9",
        ]
    )
    y_train = uncertainties["validation"][
        uncertainties["validation"]["outer_fold"] == outer_fold
    ]["entropies"]

    # substitute nan with mean
    x_train = x_train.fillna(x_train.mean())

    x_test = uncertainties["test"][
        uncertainties["test"]["outer_fold"] == outer_fold
    ].drop(
        columns=[
            "entropies",
            "mutual_informations",
            "task_name",
            "outer_fold",
            "inner_fold",
            "task_num",
            "Unnamed: 9",
        ]
    )
    x_test = x_test.fillna(x_test.mean())
    y_test = uncertainties["test"][uncertainties["test"]["outer_fold"] == outer_fold][
        "entropies"
    ]
    if not _is_fitted(model, attributes=["equations_"]):
        model.fit(x_train, y_train)
    else:
        raise NotImplementedError(
            """
                                  Model is already fitted, but I have not implemented\
                                    loading a pre-trained model. Something must have\
                                        gone wrong.
                                  """
        )

    # get train loss
    train_r2 = model.score(x_train, y_train)
    test_r2 = model.score(x_test, y_test)

    train_mse = mean_squared_error(y_train, model.predict(x_train))
    test_mse = mean_squared_error(y_test, model.predict(x_test))

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "outer_fold": outer_fold,
    }


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
    # dataset_measures_path: str = "./dataset_measures.nosync/"
    dataset_measures_path: str = configs["dataset_measures_path"]
    uncertainties_path: str = configs["uncertainties_path"]
    symbolic_regression_args: dict[str, Any] = configs["symbolic_regression_args"]
    debug_mode: bool = False

    dataset_measures = load_dataset_measures(path=dataset_measures_path)

    uncertainties = load_prepare_uncertainties(
        path=uncertainties_path, dataset_measures=dataset_measures
    )

    num_folds = get_outer_fold_num(uncertainties=uncertainties)

    if debug_mode:
        print("Debug mode is on, using only 10%% of the data")
        uncertainties = {
            key: value.sample(frac=0.1, random_state=42)
            for key, value in uncertainties.items()
        }

    results: list[dict[str, Any]] = []
    for outer_fold in tqdm(range(num_folds), desc="Outer fold progress"):
        model = prepare_symbolic_regression_model(
            outer_fold=outer_fold, **symbolic_regression_args
        )

        # TODO: I should save the symbolic regression "best result", as well as
        # a feature importance map — I thought I did it. Basically, a list of
        # how many times a feature appears in the symbolic regression task
        fold_result = train_test_symbolic_regression(
            model=model, uncertainties=uncertainties, outer_fold=outer_fold
        )
        results.append(fold_result)

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(
        os.path.join(
            symbolic_regression_args["path_to_save"], "symbolic_regression_results.csv"
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
