"""This script allows to train a specific symbolic regression for each 
combination of K and y_mean.
"""

import argparse
import hashlib
import os
import shutil
from logging import INFO, basicConfig, getLogger
from sys import path
from typing import Any

import pandas as pd
import numpy as np

path.append("./")

from src.utils.io import load_config
from experiments.verdojas_equation.run import (
    get_data_for_regression,
    prepare_symbolic_regression_model,
    add_equation_estimates_to_data,
)

logger = getLogger("run")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_name", type=str, help="Path to the config file", default="config.yaml"
)
args = parser.parse_args()


def personalized_fit(
    model,
    variables_to_personalize,
    symbolic_regression_variables,
    data_val,
    data_test,
    base_savepath,
    regression_data_val_with_equations_savepath,
    regression_data_test_with_equations_savepath,
):
    # remove variables_to_personalize from symbolic_regression_variables
    symbolic_regression_variables = [
        var
        for var in symbolic_regression_variables
        if var not in variables_to_personalize
    ]
    subsets_data_val = []
    subsets_data_test = []
    for variable_to_personalize in variables_to_personalize:
        possible_variable_values = data_val[variable_to_personalize].unique()
        for variable_value in possible_variable_values:
            subset_data_val = data_val[data_val[variable_to_personalize] == variable_value]
            subset_data_test = data_test[data_test[variable_to_personalize] == variable_value]
            model.fit(
                subset_data_val[symbolic_regression_variables].values,
                subset_data_val[["variance"]].values,
            )
            model.equations.to_csv(
                os.path.join(base_savepath, "all_formulas.csv"), index=False
            )
            model.get_best().to_csv(
                os.path.join(base_savepath, "best_formula.csv"), index=True
            )
            subset_data_val = add_equation_estimates_to_data(
                subset_data_val,
                model,
                symbolic_regression_variables=symbolic_regression_variables,
            )
            subset_data_test = add_equation_estimates_to_data(
                subset_data_test,
                model,
                symbolic_regression_variables=symbolic_regression_variables,
            )

            subset_data_val.to_csv(f"{regression_data_val_with_equations_savepath}_{variable_value}.csv")
            subset_data_test.to_csv(f"{regression_data_test_with_equations_savepath}_{variable_value}.csv")
            
            subsets_data_val.append(subset_data_val)
            subsets_data_test.append(subset_data_test)
    subsets_data_val = pd.concat(subsets_data_val)
    subsets_data_test = pd.concat(subsets_data_test)
    return subsets_data_val, subsets_data_test


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

    current_unix_time = int(np.datetime64('now').astype('datetime64[s]').astype(int))
    base_savepath: str = os.path.join(
        path_to_script_folder,
        "results_personalized.nosync",
        hashlib.md5(f"{name_experiment}_{current_unix_time}".encode()).hexdigest(),
    )
    # copy config inside base_savepath
    os.makedirs(base_savepath, exist_ok=True)
    shutil.copyfile(path_to_config, os.path.join(base_savepath, config_name))

    basicConfig(
        filename=os.path.join(base_savepath, f"{config_name.split('.')[0]}.log"),
        level=INFO,
    )
    regression_data_val, regression_data_test = get_data_for_regression(
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
        base_savepath, f"{regression_data_with_equations_filename}_val"
    )
    regression_data_test_with_equations_savepath: str = os.path.join(
        base_savepath, f"{regression_data_with_equations_filename}_test"
    )

    model = prepare_symbolic_regression_model(
        n_iterations=symoblic_regression_params["n_iterations"],
        n_population=symoblic_regression_params["n_population"],
    )

    regression_data_val, regression_data_test = personalized_fit(
        model=model,
        variables_to_personalize=["K", "y_mean"],
        symbolic_regression_variables=symbolic_regression_variables,
        data_val=regression_data_val,
        data_test=regression_data_test,
        base_savepath=base_savepath,
        regression_data_val_with_equations_savepath=regression_data_val_with_equations_savepath,
        regression_data_test_with_equations_savepath=regression_data_test_with_equations_savepath,
    )
    regression_data_val.to_csv(regression_data_val_with_equations_savepath)
    regression_data_test.to_csv(regression_data_test_with_equations_savepath)


if __name__ == "__main__":
    main()
