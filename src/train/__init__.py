import torch
import os

# import dataset and dataloader for pytoarch
import torch.utils.data
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm.auto import tqdm
from scipy.stats import mode, entropy


from src.utils.data import get_dataset
from src.model.sklearn import MLPDropout
from src.utils import OutputTypeError

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
        

def calculate_outlier_info(x_val: np.ndarray, random_seed: int = 42) -> np.ndarray:
    outlier_classifier = IsolationForest(random_state=random_seed)
    outlier_classifier.fit(x_val)
    outlier_vals = outlier_classifier.score_samples(x_val)
    return outlier_vals

def calculate_anonmaly_info(x_val: np.ndarray) -> np.ndarray:
    anomaly_classifier = LocalOutlierFactor()
    anomaly_classifier.fit(x_val)
    anomaly_vals = anomaly_classifier.score_samples(x_val)
    return anomaly_vals
    


@ignore_warnings(category=ConvergenceWarning)
def perform_fold_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    name: str,
    task_type: str,
    output_size: int,
    model_args: dict,
    train_args: dict,
    experiment_args: dict,
    random_seed: int = 42,
    num_jobs: int = -1,
):

    # mkl.set_num_threads(num_jobs)
    classifier = MLPDropout(
        random_state=random_seed,
        max_iter=train_args["num_epochs"],
        alpha=experiment_args["alpha"],
        hidden_layer_sizes=tuple(
            model_args["layer_size"] for _ in range(experiment_args["num_layers"])
        ),
        dropout=experiment_args["dropout_rate"],
        mcdropout=True,
        activation=model_args.get("hidden_activation_type", None),
        # num_jobs=num_jobs,
    )

    classifier.fit(x_train, y_train)
    y_preds_proba = [
        classifier.predict_proba(x_val) for _ in range(experiment_args["mcdropout_num"])
    ]
    y_preds = [classifier._label_binarizer.inverse_transform(prediction) for prediction in y_preds_proba]
    
    y_preds_proba = np.array(y_preds_proba)
    y_preds = np.array(y_preds)

    y_pred = mode(y_preds, axis=0)[0]
    # NOTE: we are selecting the entropy of the class chosen as "prediction"
    # entropies = np.take_along_axis(entropy(y_preds_proba, axis=0), y_pred.astype(int).reshape(-1,1), axis=1)
    # FIXME: the entropies calculation is wrong
    
    
    val_accuracy = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(
        y_val,
        y_pred,
        average="binary" if task_type == "binary classification" else "macro",
    )
    val_mcc = matthews_corrcoef(y_val, y_pred)

    outlier_vals = calculate_outlier_info(x_val=x_val, random_seed=random_seed)
    anomaly_vals = calculate_anonmaly_info(x_val=x_val, random_seed=random_seed)
    

    return {
        "task_name": name,
        "task_type": task_type,
        "output_size": output_size,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "val_mcc": val_mcc,
        "y_val_preds_proba": y_preds_proba,
        "outlier_vals": outlier_vals,
        "anomaly_vals": anomaly_vals,
        # "entropies": entropies,
    }
    
@ignore_warnings(category=ConvergenceWarning)
def perform_only_outlier_detection(x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    name: str,
    task_type: str,
    output_size: int,
    model_args: dict,
    train_args: dict,
    experiment_args: dict,
    random_seed: int = 42,
    num_jobs: int = -1,):
    
    outlier_vals = calculate_outlier_info(x_val=x_val, random_seed=random_seed)
    anomaly_vals = calculate_anonmaly_info(x_val=x_val)
    
    return {
        "task_name": name,
        "task_type": task_type,
        "output_size": output_size,
        "outlier_vals": outlier_vals,
        "anomaly_vals": anomaly_vals,
    }
    


def train(
    task_num: int,
    dataset_id: int,
    num_folds: int,
    results_path: str,
    random_seed: int,
    experiment_args: dict = {
        "dropout_rate": 0.05,
        "alpha": 0.0,
        "mcdropout_num": 10,
        "num_layers": 5,
    },
    model_args: dict = {"layer_size": 100, "hidden_activation_type": "relu"},
    train_args: dict = {"num_epochs": 100},
    num_jobs: int = -1,
    outlier_flag: bool = False,
) -> None:

    x, y, name, task_type, output_size = get_dataset(task_num=task_num)

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    kf = StratifiedKFold(n_splits=num_folds, random_state=random_seed, shuffle=True)

    fold_results: list[dict] = []
    fold = 0
    for train_index, val_index in tqdm(
        kf.split(x, y), desc="Folds", colour="magenta", total=num_folds, disable=True
    ):
        fold += 1

        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        if not outlier_flag:
            fold_result = perform_fold_prediction(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                name=name,
                task_type=task_type,
                output_size=output_size,
                model_args=model_args,
                train_args=train_args,
                experiment_args=experiment_args,
                random_seed=random_seed,
                num_jobs=num_jobs,
            )
        else:
            fold_result = perform_only_outlier_detection(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                name=name,
                task_type=task_type,
                output_size=output_size,
                model_args=model_args,
                train_args=train_args,
                experiment_args=experiment_args,
                random_seed=random_seed,
                num_jobs=num_jobs,
            )
        fold_result.update(
            {
                "fold": fold,
                "task_num": task_num,
                "experiment_args": experiment_args,
                "model_args": model_args,
                "train_args": train_args,
            }
        )
        fold_results.append(fold_result)
    # return fold_results

    output_filename: str = (
        f"task{dataset_id}_dropout_rate{experiment_args['dropout_rate']}_model_precision{experiment_args['alpha']}_num_mcdropout_iterations{experiment_args['mcdropout_num']}_num_layers{experiment_args['num_layers']}.pth"
    )
    # save list of dicts to json
    # TODO: find a better schema. Probably not a good idea to save everything at the end.
    torch.save(fold_results, os.path.join(results_path, output_filename))
