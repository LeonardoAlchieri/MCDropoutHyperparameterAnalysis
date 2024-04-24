import torch
import openml
import pandas as pd
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder
from src.utils import OutputTypeError


def prepare_prediction_array(y: ndarray) -> torch.Tensor:
    # FIXME: I think this part is unnecessary!
    # Use LabelEncoder to encode the class array
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)

    num_classes = len(set(encoded_labels))

    # # Convert the encoded labels to one-hot encoded vectors using PyTorch
    # class_array_one_hot = torch.nn.functional.one_hot(
    #     torch.tensor(encoded_labels), num_classes
    # )
    # if class_array_one_hot.shape[1] == 2:
    #     print(f"Binary classification detected. Converting to single vector")
    #     # Convert binary classification to a single vector
    #     class_array_one_hot = class_array_one_hot[:, 1].unsqueeze(1)
    return torch.tensor(encoded_labels, dtype=torch.float32), num_classes


def get_dataset(task_num: int, outer_fold_idxs: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor, str, str, int]:
    # 99 is the ID of the OpenML-CC18 study
    task = openml.tasks.get_task(task_num)
    dataset_obj = task.get_dataset()
    dataset = dataset_obj.get_data()[0]
    
    # drop nan
    dataset = dataset.dropna(how='any')

    x = dataset.drop(columns=[task.target_name])
    x = pd.get_dummies(x)
    # substitute the NaN values with the mean of the column
    x = torch.tensor(x.values.astype(float), dtype=torch.float32)

    y = dataset[task.target_name].to_numpy()
    y, num_classes = prepare_prediction_array(y)
    
    name = dataset_obj.name

    prediction_type = task.task_type
    if prediction_type == "Supervised Classification":
        prediction_type = (
            "multiclass classification" if num_classes > 1 else "binary classification"
        )
    elif "Regression" in prediction_type:
        prediction_type = "regression"
    else:
        raise ValueError(
            f"Found prediction type {prediction_type}. Support are 'Supervised Classification' and 'Regression'."
        )
    
    if outer_fold_idxs is not None:
        x = x[outer_fold_idxs]
        y = y[outer_fold_idxs]
    return x, y, name, prediction_type, num_classes


def load_dataset_subsample(file_path: str) -> list:
    with open(file_path, "r") as file:
        dataset_subsample = [(line.strip()) for line in file]

    return dataset_subsample

def get_outer_fold(
    path: str,
    fold_type: str = "train,"
) -> dict[str, dict[int, list[int]]]:
    outer_fold_info = torch.load(path, map_location=torch.device('cpu'))
    return outer_fold_info[fold_type]
    
    
    
    
    ...