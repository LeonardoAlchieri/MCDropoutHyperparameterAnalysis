import argparse
import hashlib
import os
import shutil
from logging import INFO, basicConfig, getLogger
from sys import path
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

path.append("./")

from src.train import run_dropout_estimation
from src.utils.io import load_config

logger = getLogger("run")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_name", type=str, help="Path to the config file", default="config.yaml"
)
args = parser.parse_args()


def prepare_symbolic_regression_model(
    n_iterations: int,
    n_population: int,
    random_state: int = 42,
    num_jobs: int = 6,
) -> PySRRegressor:

    save_directory: str = f"./temp_equation_files.nosync/verdoja/"
    if not os.path.exists(save_directory):
        warn(f"Creating directory {save_directory}, since it did not exist.")
        os.makedirs(save_directory)

    return PySRRegressor(
        niterations=n_iterations,  # < Increase me for better results
        populations=n_population,
        population_size=33,
        ncycles_per_iteration=550 * 2,
        maxsize=50,
        # ncycles_per_iteration=50,
        binary_operators=["+", "-", "*", "/"],
        #   "^"],
        unary_operators=[
            # "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
            "square",
            # "cube",
            # "exp",
            # "abs",
            # "log",
            # "log10",
            # "log2",
            # "log1p",
            # "sqrt",
        ],
        random_state=random_state,
        procs=num_jobs,
        multithreading=False,
        batching=True,
        temp_equation_file=True,
        tempdir=save_directory,
        delete_tempfiles=False,
        # heap_size_hint_in_bytes=int(2e10),
        precision=64,
        # turbo=True,
        # bumper=True,
        # deterministic=True,
        # equation_file=os.path.join(path_to_save, f"equation_{outer_fold}.csv"),
    )


def prepare_synthetic_data(
    dim_x: list[int], val_y: list[int]
) -> dict[tuple[int, int], dict[str, tuple[np.ndarray, np.ndarray]]]:
    num_samples_in_dataset: int = 3200 + 500 + 500

    # dim_x: list[int] = [5, 10, 100, 500, 1000, 5000, 10000, 20000]
    # dim_x: list[int] = [5, 10, 100]
    # val_y: list[int] = [1, 10, 100]
    # val_y: list[int] = [1, 10, 100, 1000]

    Xs = {dim: np.ones(shape=(num_samples_in_dataset, dim)) for dim in dim_x}
    ys = {
        val: np.random.normal(loc=val, scale=1, size=num_samples_in_dataset)
        for val in val_y
    }

    # get all combinations of x and y into a dictionary
    datasets: dict[tuple[int, int], dict[str, tuple[np.ndarray, np.ndarray]]] = {
        (i, j): {
            "train": (x[:3200], y[:3200]),
            "val": (
                x[3200 : int(3200 + 500)],
                y[3200 : int(3200 + 500)],
            ),
            "test": (
                x[int(3200 + 500) :],
                y[int(3200 + 500) :],
            ),
        }
        for i, x in Xs.items()
        for j, y in ys.items()
    }
    return datasets


def format_uncertainty_data(variances_dropout: dict) -> pd.DataFrame:
    regression_data = pd.DataFrame.from_dict(variances_dropout, orient="index")
    regression_data.index = pd.MultiIndex.from_tuples(
        regression_data.index, names=["K", "y_mean", "p"]
    )
    regression_data = regression_data.reset_index(inplace=False)
    regression_data = regression_data.rename(
        columns={0: "variance", 1: "log likelihood"}, inplace=False
    )
    return regression_data


def calculate_log_likelihood(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    tau: float,
) -> float:
    return (
        np.log(
            np.sum(np.exp(-0.5 * tau * np.sum((y_true - y_pred) ** 2, axis=1)), axis=0)
        )
        - np.log(len(y_true))
        - 0.5 * np.log(2 * np.pi)
        + 0.5 * np.log(tau)
    )


def get_tau(l2, p, alpha, N):
    return l2 * (1 - p) / (alpha * 2 * N)


def get_uncertainty_data(
    dropouts: list[float] = [0.001, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
    dim_x: list[int] = [5, 10, 100, 500, 1000, 5000, 10000, 20000],
    val_y: list[int] = [1, 10, 100, 1000],
    num_mcdropout: int = int(1e5),
    num_epochs: int = 600,
    batch_sizes: int = 16,
    l2: float = 0.0001**2,
    alpha: float = 0.0005,
    model_args: dict = {"layer_size": 64, "hidden_activation": "relu"},
    path_to_save_val: str = "uncertainty_data_val.csv",
    path_to_save_test: str = "uncertainty_data_test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    datasets = prepare_synthetic_data(dim_x=dim_x, val_y=val_y)
    variances_dropout_val: dict[tuple[int, int, float], float] = {}
    variances_dropout_test: dict[tuple[int, int, float], float] = {}

    # dropouts = [0.001, 0.05, 0.1]

    for (K, y_mean), dataset in tqdm(datasets.items(), desc="Dataset progress"):
        fn_inputs = [
            (
                dataset,
                K,
                y_mean,
                p,
                num_mcdropout,
                num_epochs,
                batch_sizes,
                alpha,
                model_args,
            )
            for p in dropouts
        ]

        fn_outputs = process_map(run_dropout_estimation, fn_inputs, max_workers=9)
        variances_dropout_val.update(
            {
                (K, y_mean, p): (
                    np.asarray(mc_predictions["val"]).var(),
                    calculate_log_likelihood(
                        mc_predictions["val"],
                        dataset["val"][1],
                        get_tau(l2, p, alpha, len(dataset["val"][1])),
                    ),
                )
                for K, y_mean, p, mc_predictions in fn_outputs
            }
        )
        variances_dropout_test.update(
            {
                (K, y_mean, p): (
                    np.asarray(mc_predictions["test"]).var(),
                    calculate_log_likelihood(
                        mc_predictions["test"],
                        dataset["test"][1],
                        get_tau(l2, p, alpha, len(dataset["test"][1])),
                    ),
                )
                for K, y_mean, p, mc_predictions in fn_outputs
            }
        )
        format_uncertainty_data(variances_dropout_val).to_csv(path_to_save_val)
        format_uncertainty_data(variances_dropout_test).to_csv(path_to_save_test)
        # variances_dropout_val[(K, y_mean, p)] = np.asarray(mc_predictions).var()
    return (
        format_uncertainty_data(variances_dropout_val),
        format_uncertainty_data(variances_dropout_test),
    )


def add_equation_estimates_to_data(
    regression_data: pd.DataFrame,
    model: PySRRegressor,
    symbolic_regression_variables: list[str],
) -> pd.DataFrame:
    regression_data["variance_verdoja"] = regression_data[["K", "y_mean", "p"]].apply(
        lambda x: (x[0] * x[2] * (1 - x[2]) * (x[1] ** 2))
        / ((x[0] - x[0] * x[2] + x[2]) ** 2),
        axis=1,
    )
    regression_data["our_formula"] = regression_data[
        symbolic_regression_variables
    ].apply(
        lambda x: model.predict(x.values.reshape(1, -1))[0],
        # (x[1])**2 / (((x[0] + (1.8603)**2) / x[2]) - x[0]),
        axis=1,
    )
    return regression_data


def get_data_for_regression(
    base_savepath: str,
    regression_data_filename: str,
    regression_data_with_equations_filename: str,
    dropouts: list[float],
    dim_input_data: list[int],
    val_y_mean: list[int],
    num_mcdropout: int,
    num_epochs: int,
    batch_sizes: int,
    l2: float,
    alpha: float,
    model_args: dict[str, Any],
    alternative_path_to_saved_data: str | None = None,
    recreate_data: bool = False,
):
    regression_data_savepath: str = os.path.join(
        base_savepath, regression_data_filename
    )
    regression_data_savepath_val = f"{regression_data_savepath}_val.csv"
    regression_data_savepath_test = f"{regression_data_savepath}_test.csv"

    
    if alternative_path_to_saved_data:
        alternative_path_to_saved_data_val = os.path.join(
            alternative_path_to_saved_data,
            f"{regression_data_with_equations_filename}_val.csv",
        )
        alternative_path_to_saved_data_test = os.path.join(
            alternative_path_to_saved_data,
            f"{regression_data_with_equations_filename}_test.csv",
        )

    # dataset_measures_path: str = "./dataset_measures.nosync/"

    if (
        not os.path.isfile(regression_data_savepath_val)
        and not os.path.isfile(regression_data_savepath_test)
    ) or recreate_data:
        print(f"Creating regression data at {regression_data_savepath}")
        os.makedirs(os.path.dirname(regression_data_savepath_val), exist_ok=True)
        os.makedirs(os.path.dirname(regression_data_savepath_test), exist_ok=True)
        if not alternative_path_to_saved_data:
            regression_data_val, regression_data_test = get_uncertainty_data(
                dropouts=dropouts,
                dim_x=dim_input_data,
                val_y=val_y_mean,
                num_mcdropout=num_mcdropout,
                num_epochs=num_epochs,
                batch_sizes=batch_sizes,
                l2=l2,
                alpha=alpha,
                model_args=model_args,
                path_to_save_val=regression_data_savepath_val,
                path_to_save_test=regression_data_savepath_test,
            )
        else:
            print("!!!! Alternative path to saved data provided. Reading from it. !!!!")
            regression_data_val = pd.read_csv(
                alternative_path_to_saved_data_val, index_col=0
            )
            regression_data_test = pd.read_csv(
                alternative_path_to_saved_data_test, index_col=0
            )
        regression_data_val.to_csv(regression_data_savepath_val)
        regression_data_test.to_csv(regression_data_savepath_test)
    else:
        print(
            f"Loading regression data from {regression_data_savepath_val} and {regression_data_savepath_test}, because already present."
        )
        regression_data_val = pd.read_csv(regression_data_savepath_val, index_col=0)
        regression_data_test = pd.read_csv(regression_data_savepath_test, index_col=0)
    return regression_data_val, regression_data_test


def main():
    path_to_script_folder: str = os.path.dirname(os.path.abspath(__file__))
    config_name: str = args.config_name

    path_to_config: str = os.path.join(path_to_script_folder, config_name)

    configs: dict[str, Any] = load_config(path=path_to_config)
    regression_data_filename: str = configs["regression_data_filename"]
    regression_data_with_equations_filename = configs[
        "regression_data_with_equations_filename"
    ]
    name_experiment: str = configs["name_experiment"]
    dropouts: list[float] = configs["dropouts"]
    dim_input_data: list[int] = configs["dim_input_data"]
    val_y_mean: list[int] = configs["val_y_mean"]
    num_mcdropout: int = configs["num_mcdropout"]
    num_epochs: int = configs["num_epochs"]
    batch_sizes: int = configs["batch_sizes"]
    l2: float = configs["l2"]
    alpha: float = configs["alpha"]
    model_args: dict[str, Any] = configs["model_args"]
    symbolic_regression_variables: list[str] = configs["symbolic_regression_variables"]
    symoblic_regression_params: dict[str, Any] = configs["symoblic_regression_params"]
    recreate_data: bool = configs["recreate_data"]

    alternative_path_to_saved_data: str | None = configs.get(
        "alternative_path_to_saved_data", None
    )

    # get current unix time
    current_unix_time = int(np.datetime64('now').astype('datetime64[s]').astype(int))
    
    
    base_savepath: str = os.path.join(
        path_to_script_folder,
        "results.nosync",
        hashlib.md5(f"{name_experiment}_{current_unix_time}".encode()).hexdigest(),
    )
    # copy config inside base_savepath
    os.makedirs(base_savepath, exist_ok=True)
    shutil.copyfile(path_to_config, os.path.join(base_savepath, config_name))

    basicConfig(
        filename=os.path.join(base_savepath, f"{config_name.split('.')[0]}.log"),
        level=INFO,
    )
    regression_data_val, regression_data_test  = get_data_for_regression(
        base_savepath=base_savepath,
        regression_data_filename=regression_data_filename,
        regression_data_with_equations_filename=regression_data_with_equations_filename,
        dropouts=dropouts,
        dim_input_data=dim_input_data,
        val_y_mean=val_y_mean,
        num_mcdropout=num_mcdropout,
        num_epochs=num_epochs,
        batch_sizes=batch_sizes,
        l2=l2,
        alpha=alpha,
        model_args=model_args,
        alternative_path_to_saved_data=alternative_path_to_saved_data,
        recreate_data=recreate_data,
    )
    regression_data_val_with_equations_savepath: str = os.path.join(
        base_savepath, f"{regression_data_with_equations_filename}_val.csv"
    )
    regression_data_test_with_equations_savepath: str = os.path.join(
        base_savepath, f"{regression_data_with_equations_filename}_test.csv"
    )

    model = prepare_symbolic_regression_model(
        n_iterations=symoblic_regression_params["n_iterations"],
        n_population=symoblic_regression_params["n_population"],
    )
    model.fit(
        regression_data_val[symbolic_regression_variables].values,
        regression_data_val[["variance"]].values,
    )
    # print best pysr formula to a file
    model.equations.to_csv(os.path.join(base_savepath, "all_formulas.csv"), index=False)
    model.get_best().to_csv(os.path.join(base_savepath, "best_formula.csv"), index=True)
    regression_data_val = add_equation_estimates_to_data(
        regression_data_val,
        model,
        symbolic_regression_variables=symbolic_regression_variables,
    )
    regression_data_test = add_equation_estimates_to_data(
        regression_data_test,
        model,
        symbolic_regression_variables=symbolic_regression_variables,
    )

    regression_data_val.to_csv(regression_data_val_with_equations_savepath)
    regression_data_test.to_csv(regression_data_test_with_equations_savepath)


if __name__ == "__main__":
    main()
