from typing import Any
from yaml import safe_load as load_yaml
import pandas as pd
from glob import glob
import torch
from tqdm.auto import tqdm
import os

def load_config(path: str) -> dict[str, Any]:
    """Simple method to load yaml configuration for a given script.

    Args:
        path (str): path to the yaml file

    Returns:
        Dict[str, Any]: the method returns a dictionary with the loaded configurations
    """
        
    with open(path, "r") as file:
        config_params = load_yaml(file)
    return config_params
    



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