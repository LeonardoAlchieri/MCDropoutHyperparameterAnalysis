import argparse
import os
from gc import collect as pick_up_trash
from glob import glob
from logging import INFO, basicConfig, getLogger
from sys import path
from warnings import warn
from time import time
from typing import Any, Literal

from sklearn.utils._testing import ignore_warnings
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm.contrib.concurrent import process_map
from tqdm.auto import tqdm
import psutil
import ctypes


path.append("./")

from src.utils import OutputTypeError
from src.utils.io import load_config

logger = getLogger("run")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_name", type=str, help="Path to the config file", default="config.yaml"
)
args = parser.parse_args()


def prepare_dict_for_regression(path_to_result: str, fold_id: int = 1) -> dict:
    # FIXME: torch documentation suggests to avoid using weights_only=False
    # for safety reasons. However, I cannot load with weights_only=True
    loaded_dict_base = torch.load(
        path_to_result, map_location=torch.device("cpu"), weights_only=False
    )
    if len(loaded_dict_base[fold_id]) == 0:
        return None

    return loaded_dict_base


def custom_entropy_formula(predictions: np.array) -> np.array:
    return -np.nansum(
        np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0)), axis=1
    ) / np.log(predictions.shape[2])


def custom_variational_ratios_formula(predictions: np.array) -> np.array:
    predicted_classes = np.argmax(predictions, axis=2)
    frequency_predicted_classes = np.array(
        [
            np.bincount(predicted_classes[:, i], minlength=predictions.shape[2])
            for i in range(predictions.shape[1])
        ]
    )
    frequency_predicted_classes_argmax = np.argmax(frequency_predicted_classes, axis=1)
    return (
        1
        - np.take_along_axis(
            frequency_predicted_classes,
            frequency_predicted_classes_argmax[:, None],
            axis=1,
        )
        / predictions.shape[0]
    )


def custom_mutual_information_formula(
    predictions: np.array, entropies: np.array
) -> np.array:
    average_entropy = np.mean(
        -np.nansum(predictions * np.log(predictions), axis=2), axis=0
    ) / np.log(predictions.shape[2])
    return average_entropy - entropies


def calculate_classification_uncertainties(
    df: pd.DataFrame, predictions: np.ndarray, validation: bool = True
) -> pd.DataFrame:
    entropies = custom_entropy_formula(predictions)
    # variational_ratios = custom_variational_ratios_formula(predictions)
    mutual_informations = custom_mutual_information_formula(predictions, entropies)

    return pd.DataFrame.from_dict(
        {
            "entropies": entropies,
            # "variational_ratios": variational_ratios[0],
            "mutual_informations": mutual_informations,
            # "y_preds_proba": df['y_preds_proba'].values[0],
            "outlier_vals": (
                df["val_outlier_vals"].values[0]
                if validation
                else df["test_outlier_vals"].values[0]
            ),
            "anomaly_vals": (
                df["val_anomaly_vals_test"].values[0]
                if validation
                else df["test_anomaly_vals_test"].values[0]
            ),
        },
        orient="index",
    ).T


def calculate_regression_uncertainties(
    df: pd.DataFrame, predictions: np.ndarray, validation: bool = True
) -> pd.DataFrame:
    variances = np.var(predictions, axis=0)
    return pd.DataFrame.from_dict(
        {
            "variances": variances,
            "outlier_vals": (
                df["val_outlier_vals"].values[0]
                if validation
                else df["test_outlier_vals"].values[0]
            ),
            "anomaly_vals": (
                df["val_anomaly_vals_test"].values[0]
                if validation
                else df["test_anomaly_vals_test"].values[0]
            ),
        },
        orient="index",
    ).T


def calculate_uncertainties(
    df: pd.DataFrame,
    validation: bool = True,
) -> pd.DataFrame:
    if len(df) > 1:
        raise ValueError("More than one result in the group. This should not happen.")

    task: Literal["classification", "regression"] = df["task_type"].iloc[0]

    if task == "classification":
        pred_name: str = "y_val_preds_proba" if validation else "y_test_preds_proba"
    elif task == "regression":
        pred_name: str = "y_val_preds" if validation else "y_test_preds"
    else:
        raise OutputTypeError("task", task, ["classification", "regression"])

    predictions = df[pred_name].iloc[0]

    return (
        calculate_classification_uncertainties(df, predictions, validation)
        if task == "classification"
        else calculate_regression_uncertainties(df, predictions, validation)
    )


def prepare_result_df(loaded_data: pd.DataFrame, idx: int) -> pd.DataFrame | None:
    if loaded_data is None:
        return None
    loaded_data = [
        pd.DataFrame.from_dict(res, orient="index", columns=[(idx * 3) + (i)]).T
        for i, res in enumerate(loaded_data)
        if res is not None
    ]
    if len(loaded_data) < 3:
        warn(f"Less than 3 folds for {path}. Skipping.", RuntimeWarning)
        return None
    path_results = pd.concat(
        loaded_data, keys=["fold1", "fold2", "fold3"], names=["fold"]
    )
    path_results = path_results.set_index(["outer_fold", "inner_fold"], inplace=False)
    path_results = path_results.sort_index(inplace=False)

    path_results.index = pd.MultiIndex.from_tuples(
        [
            (outer_fold, inner_fold, i)
            for i, (outer_fold, inner_fold) in enumerate(path_results.index)
        ]
    )
    return path_results


def cleanup_results(data: pd.DataFrame) -> pd.DataFrame:
    data["alpha"] = data["experiment_args"].apply(lambda x: x["alpha"])
    data["mcdropout_num"] = data["experiment_args"].apply(lambda x: x["mcdropout_num"])
    data["num_layers"] = data["experiment_args"].apply(lambda x: x["num_layers"])
    data["dropout_rate"] = data["experiment_args"].apply(lambda x: x["dropout_rate"])
    data["layer_size"] = data["model_args"].apply(lambda x: x["layer_size"])
    data["hidden_activation_type"] = data["model_args"].apply(
        lambda x: x["hidden_activation_type"]
    )
    data.drop(columns=["experiment_args", "model_args", "train_args"], inplace=True)
    data.index.names = ["outer_fold", "inner_fold", "idx"]
    data = data.reset_index(drop=False, inplace=False)
    return data


@ignore_warnings(category=DeprecationWarning)
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
    path_to_mlp_results = configs["path_to_mlp_results"]
    path_for_save_validation_data: str = configs["path_for_save_validation_data"]
    path_for_save_test_data: str = configs["path_for_save_test_data"]
    ram_limit: int = configs["ram_limit"]
    # n_jobs = configs["num_jobs"]

    all_results_path = glob(path_to_mlp_results + "*.pth")

    current_os = os.uname().sysname
    if current_os == "Darwin":
        libc = ctypes.CDLL("libSystem.dylib")
    elif current_os == "Linux":
        libc = ctypes.CDLL("libc.so.6")
    else:
        raise OSError("Unsupported OS. Only MacOS and Linux are supported. Received %s" % current_os)
    idx = 0
    for path in (
        pbar := tqdm(
            all_results_path,
            total=len(all_results_path),
            desc=("Loading data"),
            smoothing=0,
        )
    ):
        try:
            loaded_data = prepare_dict_for_regression(path, 1)
        except RuntimeError as e:
            warn(f"Error while loading data: {e}. Skipping this file.")
            logger.error(f"Skipping file {path} due to error: {e}")
            continue

        path_results = prepare_result_df(loaded_data, idx)
        if path_results is None:
            continue

        path_results = cleanup_results(path_results)
        val_result = path_results.groupby(
            [
                "outer_fold",
                "inner_fold",
                "task_name",
                "task_num",
                "alpha",
                "mcdropout_num",
                "num_layers",
                "dropout_rate",
                "output_size",
            ]
        ).apply(calculate_uncertainties, validation=True)

        test_result = path_results.groupby(
            [
                "outer_fold",
                "inner_fold",
                "task_name",
                "task_num",
                "alpha",
                "mcdropout_num",
                "num_layers",
                "dropout_rate",
                "output_size",
            ],
        ).apply(calculate_uncertainties, validation=False)

        val_result.to_csv(
            path_for_save_validation_data,
            mode="a" if idx > 0 else "w",
            header=False if idx > 0 else True,
        )
        test_result.to_csv(
            path_for_save_test_data,
            mode="a" if idx > 0 else "w",
            header=False if idx > 0 else True,
        )

        
        if current_os == "Darwin":
            libc.malloc_zone_pressure_relief(0)
        else:
            libc.malloc_trim(0)
        
        del path_results
        del loaded_data
        del val_result
        del test_result
        pick_up_trash()
        
        idx += 3


    # val_uncertainties_results = pd.concat(val_results)
    # test_uncertainties_results = pd.concat(test_results)

    # start_saving = time()
    # val_uncertainties_results.to_parquet(path_for_save_validation_data)
    # print(f'Saving validation data took {time() - start_saving} s')
    # test_uncertainties_results.to_parquet(path_for_save_test_data)
    # print(f'Saving test data took {time() - start_saving} s')


if __name__ == "__main__":
    main()
