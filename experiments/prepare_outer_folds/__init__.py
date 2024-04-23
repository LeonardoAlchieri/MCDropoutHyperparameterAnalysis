from typing import Any
import argparse
import torch
import os
from sys import path
from logging import getLogger
from tqdm.auto import tqdm
import json
from sklearn.model_selection import StratifiedKFold

path.append("./")

from src.utils.data import load_dataset_subsample
from src.utils.io import load_config
from src.utils.data import get_dataset
from src.utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--error_handling", type=str, help="Error handling method", default="ignore"
)
parser.add_argument(
    "--config_name", type=str, help="Path to the config file", default="config.yaml"
)
args = parser.parse_args()


logger = getLogger("main")

def main():
    path_to_script_folder: str = os.path.dirname(os.path.abspath(__file__))
    config_name: str = args.config_name
    
    set_logger(path_to_script_folder=path_to_script_folder, config_name=config_name)
    
    path_to_config: str = os.path.join(path_to_script_folder, config_name)
    configs: dict[str, Any] = load_config(path=path_to_config)
    
    subsample_path: str = configs["subsample_path"]
    random_seed: int = configs["random_seed"]
    n_outer_folds: int = configs["n_outer_folds"]
    outer_fold_info_path: str = configs["outer_fold_info_path"]
    
    datasets_to_use = load_dataset_subsample(subsample_path)
    train_idxs_s = {}
    test_idxs_s = {}
    for task_num in tqdm(datasets_to_use, desc='Dataset progress'):
        x, y, name, task_type, output_size = get_dataset(
            task_num=task_num, outer_fold_idxs=None
        )
        clf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_seed)
        train_idxs = {outer_fold: train_idx for outer_fold, (train_idx, _) in enumerate(clf.split(x, y))}
        test_idxs = {outer_fold: test_idx for outer_fold, (_, test_idx) in enumerate(clf.split(x, y))}
        
        train_idxs_s[task_num] = train_idxs
        test_idxs_s[task_num] = test_idxs
    
    outer_fold_info = {
        "train": train_idxs_s,
        "test": test_idxs_s,
    }
    # save outer_fold_info to json
    torch.save(outer_fold_info, os.path.join(outer_fold_info_path))
    
        
    
if __name__ == "__main__":
    main()