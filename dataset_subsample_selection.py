import random

import numpy as np
import openml
import pandas as pd
from tqdm.auto import tqdm

random_seed: int = 99

n_subsample: int = 20

random.seed(random_seed)


def main():
    # 99 is the ID of the OpenML-CC18 study
    suite = openml.study.get_suite(99)
    tasks = {task: openml.tasks.get_task(task).get_dataset() for task in tqdm(suite.tasks)}
    tasks = {
        key: task.qualities["NumberOfInstances"] / len(task.features)
        for key, task in tasks.items()
    }

    df = pd.DataFrame({"instance_features_ratio": tasks})
    third_quartile = np.percentile(df["instance_features_ratio"], 75)
    tasks_to_keep = [task for task in tasks.keys() if tasks[task] < third_quartile]

    sampled_tasks = random.sample(tasks_to_keep, 20)

    # save list to csv
    with open("subsampled_tasks.csv", "w") as f:
        for task in sampled_tasks:
            f.write("%s\n" % task)


if __name__ == "__main__":
    main()
