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

path.append("./")

from src.utils.io import load_config

logger = getLogger("run")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_name", type=str, help="Path to the config file", default="config.yaml"
)
args = parser.parse_args()


def load_dataset_measures(path: str) -> pd.DataFrame:
    all_dataset_measures_paths = glob(os.path.join(path, "*.pth"))

    dataset_measures = [
        torch.load(path, map_location=torch.device("cpu"))
        for path in tqdm(all_dataset_measures_paths)
    ]
    dataset_measures = pd.DataFrame.from_dict(dataset_measures)
    dataset_measures = dataset_measures.set_index("task_name")
    dataset_measures = dataset_measures[
        [
            "dimensionality",
            "intrinsic_dim",
            "intrinsic_dim_ratio",
            "feature_noise",
            "levene_stat_avg",
            "levene_pval_avg",
            "levene_success_ratio",
            "fcc_mean",
            "skew_mean",
            "kurtosis_mean",
            "mi_mean",
            "imbalance_ratio",
        ]
    ]
    return dataset_measures


def add_dataset_measures_to_uncertainties(
    dataset_measures: pd.DataFrame, uncertainties_df: pd.DataFrame
) -> pd.DataFrame:
    for col in tqdm(dataset_measures.columns, desc="Measure progress"):
        # uncertainties_results[col] = uncertainties_results['task_name'].parallel_apply(lambda x: dataset_measures.loc[x, col])
        uncertainties_df[col] = None
        for task_name in tqdm(
            uncertainties_df["task_name"].unique(),
            desc="Dataset progress",
            disable=True,
        ):
            uncertainties_df.loc[uncertainties_df["task_name"] == task_name, col] = (
                dataset_measures.loc[task_name, col]
            )

    return uncertainties_df


def load_prepare_uncertainties(
    path: str, dataset_measures: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    uncertainties_path: list[str] = glob(os.path.join(path, "uncertainties_*.csv"))

    uncertainties: dict[str, pd.DataFrame] = {
        path.split("/")[-1].split("_")[-1].split(".")[0]: pd.read_csv(path)
        for path in uncertainties_path
    }

    uncertainties = {
        key: add_dataset_measures_to_uncertainties(
            dataset_measures=dataset_measures, uncertainties_df=uncertainties_df
        )
        for key, uncertainties_df in uncertainties.items()
    }

    return uncertainties


def prepare_symbolic_regression_model(
    outer_fold: int,
    n_iterations: int,
    n_population: int,
    path_to_save: str,
    random_state: int = 42,
    num_jobs: int = -1,
) -> PySRRegressor:
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
        tempdir="./temp_equation_files.nosync",
        delete_tempfiles=True,
        equation_file=os.path.join(path_to_save, f"equation_{outer_fold}.csv"),
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

    model.fit(x_train, y_train)

    # get train loss
    train_loss = model.score(x_train, y_train)
    test_loss = model.score(x_test, y_test)

    return {"train_loss": train_loss, "test_loss": test_loss, "outer_fold": outer_fold}


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
        ("Debug mode is on, using only 10\% of the data")
        uncertainties = {
            key: value.sample(frac=0.1, random_state=42)
            for key, value in uncertainties.items()
        }

    results: list[dict[str, Any]] = []
    for outer_fold in tqdm(range(num_folds), desc="Outer fold progress"):
        model = prepare_symbolic_regression_model(
            outer_fold=outer_fold, **symbolic_regression_args
        )

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
