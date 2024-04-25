import argparse
import os
from gc import collect as pick_up_trash
from glob import glob
from logging import INFO, basicConfig, getLogger
from sys import path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pandarallel import pandarallel
from tqdm.auto import tqdm

path.append("./")

from src.utils import OutputTypeError
from src.utils.io import load_config

logger = getLogger("run")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_name", type=str, help="Path to the config file", default="config.yaml"
)
args = parser.parse_args()


def prepare_dict_for_regression(path_to_result: str, fold_id: int) -> dict:
    loaded_dict_base = torch.load(path_to_result, map_location=torch.device("cpu"))
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


def calcualte_uncertainties(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > 1:
        raise ValueError("More than one result in the group. This should not happen.")

    
    val_predictions = df["y_val_preds_proba"].iloc[0]
    val_entropies = custom_entropy_formula(val_predictions)
    val_variational_ratios = custom_variational_ratios_formula(val_predictions)
    val_mutual_informations = custom_mutual_information_formula(val_predictions, val_entropies)
    
    test_predictions = df["y_test_preds_proba"].iloc[0]
    test_entropies = custom_entropy_formula(test_predictions)
    test_variational_ratios = custom_variational_ratios_formula(test_predictions)
    test_mutual_informations = custom_mutual_information_formula(test_predictions, test_entropies)
    
    return pd.DataFrame.from_dict(
        {
            "val_entropies": val_entropies,
            "test_entropies": test_entropies,
            "val_variational_ratios": val_variational_ratios[0],
            "test_variational_ratios": test_variational_ratios[0],
            "val_mutual_informations": val_mutual_informations,
            "test_mutual_informations": test_mutual_informations,
            # "y_val_preds_proba": df['y_val_preds_proba'].values[0],
            "val_outlier_vals": df["val_outlier_vals"].values[0],
            "test_outlier_vals": df["test_outlier_vals"].values[0],
            "test_anomaly_vals": df["test_anomaly_vals_test"].values[0],
            "test_anomaly_vals": df["test_anomaly_vals_test"].values[0],
        },
        orient="index",
    ).T


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
    path_to_save_data = configs["path_to_save_data"]
    n_jobs = configs["num_jobs"]
    
    # pandarallel.initialize(progress_bar=True, nb_workers=8)
    tqdm.pandas()

    all_results_path = glob(path_to_mlp_results + "*.pth")

    all_results_all_folds = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(prepare_dict_for_regression)(path, 1)
        for path in tqdm(all_results_path, total=len(all_results_path))
    )
    all_results_fold1 = [item[0] for item in all_results_all_folds if item is not None]
    all_results_fold2 = [item[1] for item in all_results_all_folds if item is not None]
    all_results_fold3 = [item[2] for item in all_results_all_folds if item is not None]
    del all_results_all_folds

    pick_up_trash()

    all_results_fold1_df = [res for res in all_results_fold1 if res is not None]
    all_results_fold2_df = [res for res in all_results_fold2 if res is not None]
    all_results_fold3_df = [res for res in all_results_fold3 if res is not None]

    all_results_fold1_df = pd.DataFrame(all_results_fold1_df)
    all_results_fold2_df = pd.DataFrame(all_results_fold2_df)
    all_results_fold3_df = pd.DataFrame(all_results_fold3_df)

    # concat the three datasets, using a new index called "fold"
    all_results = pd.concat(
        [all_results_fold1_df, all_results_fold2_df, all_results_fold3_df],
        keys=["fold1", "fold2", "fold3"],
        names=["fold"],
    )
    del all_results_fold1_df
    del all_results_fold2_df
    del all_results_fold3_df

    all_results = all_results.set_index(["outer_fold", "inner_fold"], inplace=False)
    all_results = all_results.sort_index(inplace=False)

    all_results.index = pd.MultiIndex.from_tuples(
        [
            (outer_fold, inner_fold, i)
            for i, (outer_fold, inner_fold) in enumerate(all_results.index)
        ]
    )

    pick_up_trash()
    all_results["alpha"] = all_results["experiment_args"].apply(lambda x: x["alpha"])
    all_results["mcdropout_num"] = all_results["experiment_args"].apply(
        lambda x: x["mcdropout_num"]
    )
    all_results["num_layers"] = all_results["experiment_args"].apply(
        lambda x: x["num_layers"]
    )
    all_results["dropout_rate"] = all_results["experiment_args"].apply(
        lambda x: x["dropout_rate"]
    )
    all_results["layer_size"] = all_results["model_args"].apply(
        lambda x: x["layer_size"]
    )
    all_results["hidden_activation_type"] = all_results["model_args"].apply(
        lambda x: x["hidden_activation_type"]
    )
    all_results.drop(
        columns=["experiment_args", "model_args", "train_args"], inplace=True
    )
    all_results.index.names = ["outer_fold", "inner_fold", "idx"]
    all_results = all_results.reset_index(drop=False, inplace=False)

    uncertainties_results: pd.DataFrame = all_results.groupby(
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
    ).progress_apply(calcualte_uncertainties)

    uncertainties_results.to_csv(path_to_save_data)


if __name__ == "__main__":
    main()