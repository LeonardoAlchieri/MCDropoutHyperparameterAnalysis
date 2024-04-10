import random

import numpy as np
import openml
import pandas as pd
from tqdm.auto import tqdm

random_seed: int = 99

n_subsample: int = 20

ratio_size: int = 50
ratio_ratio: int = 75

random.seed(random_seed)


def main():
    # 99 is the ID of the OpenML-CC18 study
    suite = openml.study.get_suite(99)
    tasks = {
        task: openml.tasks.get_task(task).get_dataset() for task in tqdm(suite.tasks)
    }
    tasks = {
        key: {
            "number_of_instances": task.qualities["NumberOfInstances"],
            "number_of_features": len(task.features),
        }
        for key, task in tasks.items()
    }

    df = pd.DataFrame.from_dict(tasks, orient="index")
    df["features_instances_ratio"] = (
        df["number_of_features"] / df["number_of_instances"]
    )

    quartile_size = np.percentile(df["number_of_instances"], ratio_size)
    tasks_to_keep = [
        task
        for task in tasks.keys()
        if tasks[task]["number_of_instances"] < quartile_size
    ]

    quartile_ratio = np.percentile(df["features_instances_ratio"], ratio_ratio)
    tasks_to_keep = [
        task
        for task in tasks_to_keep
        if tasks[task]["number_of_features"] / tasks[task]["number_of_instances"]
        < quartile_ratio
    ]

    if len(tasks_to_keep) < n_subsample:
        raise ValueError(
            f"Only {len(tasks_to_keep)} tasks available, cannot subsample {n_subsample} tasks."
        )

    sampled_tasks = random.sample(tasks_to_keep, n_subsample)

    # save list to csv
    with open("subsampled_tasks.csv", "w") as f:
        for task in sampled_tasks:
            f.write("%s\n" % task)


if __name__ == "__main__":
    main()
