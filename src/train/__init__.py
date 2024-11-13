import os
from warnings import warn

import numpy as np
import torch

# import dataset and dataloader for pytoarch
import torch.utils.data
from scipy.stats import entropy, mode
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils._testing import ignore_warnings
from tqdm.auto import tqdm

from src.model.sklearn import MLPDropoutClassifier, MLPDropoutRegressor
from src.utils import OutputTypeError
from src.utils.data import get_dataset


def calculate_outlier_info(x: np.ndarray, random_seed: int = 42) -> np.ndarray:
    outlier_classifier = IsolationForest(random_state=random_seed)
    outlier_classifier.fit(x)
    outlier_vals = outlier_classifier.score_samples(x)
    return outlier_vals


def calculate_anonmaly_info(x: np.ndarray) -> np.ndarray:
    anomaly_classifier = LocalOutlierFactor()
    anomaly_classifier.fit(x)
    anomaly_vals = anomaly_classifier.negative_outlier_factor_
    return anomaly_vals


def get_prediction_metrics(
    classifier: MLPDropoutClassifier,
    x: np.ndarray,
    y: np.ndarray,
    task_type: str,
    experiment_args: dict,
    random_seed: int = 42,
):
    # FIXME: figure out what to do when the task is a regression. I should be calculating regression
    # values here
    y_preds_proba = [
        classifier.predict_proba(x) for _ in range(experiment_args["mcdropout_num"])
    ]
    y_preds = [
        classifier._label_binarizer.inverse_transform(prediction)
        for prediction in y_preds_proba
    ]
    y_preds_proba = np.array(y_preds_proba)
    y_preds = np.array(y_preds)

    y_pred = mode(y_preds, axis=0)[0]
    # NOTE: we are selecting the entropy of the class chosen as "prediction"
    # entropies = np.take_along_axis(entropy(y_preds_proba, axis=0), y_pred.astype(int).reshape(-1,1), axis=1)
    # FIXME: the entropies calculation is wrong

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(
        y,
        y_pred,
        average="binary" if task_type == "binary classification" else "macro",
    )
    mcc = matthews_corrcoef(y, y_pred)

    outlier_vals = calculate_outlier_info(x=x, random_seed=random_seed)
    anomaly_vals_test = calculate_anonmaly_info(x=x)
    return accuracy, f1, mcc, y_preds_proba, outlier_vals, anomaly_vals_test


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
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    random_seed: int = 42,
    batch_size: int = 200,
    num_jobs: int = -1,
):
    if task_type == "classification":
        raise NotImplementedError(
            "Classification has to be double checked. Do not try to run now"
        )
        classifier = MLPDropoutClassifier(
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
        # classifier.fit(x_train, y_train)
        # (
        #     val_accuracy,
        #     val_f1,
        #     val_mcc,
        #     y_val_preds_proba,
        #     val_outlier_vals,
        #     val_anomaly_vals_test,
        # ) = get_prediction_metrics(
        #     classifier=classifier,
        #     x=x_val,
        #     y=y_val,
        #     task_type=task_type,
        #     experiment_args=experiment_args,
        #     random_seed=random_seed,
        # )
        # if x_test is not None and y_test is not None:
        #     (
        #         test_accuracy,
        #         test_f1,
        #         test_mcc,
        #         y_test_preds_proba,
        #         test_outlier_vals,
        #         test_anomaly_vals_test,
        #     ) = get_prediction_metrics(
        #         classifier=classifier,
        #         x=x_test,
        #         y=y_test,
        #         task_type=task_type,
        #         experiment_args=experiment_args,
        #         random_seed=random_seed,
        #     )
        #     return {
        #         "task_name": name,
        #         "task_type": task_type,
        #         "output_size": output_size,
        #         "val_accuracy": val_accuracy,
        #         "val_f1": val_f1,
        #         "val_mcc": val_mcc,
        #         "y_val_preds_proba": y_val_preds_proba,
        #         "val_outlier_vals": val_outlier_vals,
        #         "val_anomaly_vals_test": val_anomaly_vals_test,
        #         "test_accuracy": test_accuracy,
        #         "test_f1": test_f1,
        #         "test_mcc": test_mcc,
        #         "y_test_preds_proba": y_test_preds_proba,
        #         "test_outlier_vals": test_outlier_vals,
        #         "test_anomaly_vals_test": test_anomaly_vals_test,
        #         # "entropies": entropies,
        #     }
        # else:
        #     return {
        #         "task_name": name,
        #         "task_type": task_type,
        #         "output_size": output_size,
        #         "val_accuracy": val_accuracy,
        #         "val_f1": val_f1,
        #         "val_mcc": val_mcc,
        #         "y_val_preds_proba": y_val_preds_proba,
        #         "val_outlier_vals": val_outlier_vals,
        #         "val_anomaly_vals_test": val_anomaly_vals_test,
        #         # "entropies": entropies,
        #     }
    elif task_type == "regression":
        classifier = MLPDropoutRegressor(
            random_state=random_seed,
            max_iter=train_args["num_epochs"],
            alpha=experiment_args["alpha"],
            hidden_layer_sizes=tuple(
                model_args["layer_size"] for _ in range(experiment_args["num_layers"])
            ),
            dropout=experiment_args["dropout_rate"],
            mcdropout=True,
            batch_size=batch_size,
            dropout_input=experiment_args.get("dropout_input", False),
            activation=model_args.get("hidden_activation_type", None),
            # epsilon = 1e-8,
            n_iter_no_change=train_args["num_epochs"],
            early_stopping=True,
            validation_fraction=0.1,
            # num_jobs=num_jobs,
        )

    classifier.fit(x_train, y_train)
    # warn(
    #     "METHOD NOT FINISHED TO IMPLEMENT. PLEASE CALCULATE PREDICTION METRICS FOR REGRESSION."
    # )
    return classifier


@ignore_warnings(category=ConvergenceWarning)
def perform_only_outlier_detection(
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
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    random_seed: int = 42,
    num_jobs: int = -1,
):

    val_outlier_vals = calculate_outlier_info(x_val=x_val, random_seed=random_seed)
    val_anomaly_vals = calculate_anonmaly_info(x_val=x_val)

    if x_test is not None and y_test is not None:
        test_outlier_vals = calculate_outlier_info(x_test, random_seed)
        test_anomaly_vals = calculate_anonmaly_info(x_test)
        return {
            "task_name": name,
            "task_type": task_type,
            "output_size": output_size,
            "val_outlier_vals": val_outlier_vals,
            "val_anomaly_vals": val_anomaly_vals,
            "test_outlier_vals": test_outlier_vals,
            "test_anomaly_vals": test_anomaly_vals,
        }
    else:
        return {
            "task_name": name,
            "task_type": task_type,
            "output_size": output_size,
            "val_outlier_vals": val_outlier_vals,
            "val_anomaly_vals": val_anomaly_vals,
        }


def train(
    task_num: int,
    dataset_id: int,
    num_inner_folds: int,
    results_path: str,
    random_seed: int,
    outer_fold_idxs_train_val: list[int] | None = None,
    outer_fold_id: str | None = None,
    outer_fold_idxs_test: list[int] | None = None,
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

    if outer_fold_id is not None:
        x_train_val = x[outer_fold_idxs_train_val]
        y_train_val = y[outer_fold_idxs_train_val]
        x_test = x[outer_fold_idxs_test]
        y_test = y[outer_fold_idxs_test]
        del x, y

    kf = StratifiedKFold(
        n_splits=num_inner_folds, random_state=random_seed, shuffle=True
    )

    inner_fold_results: list[dict] = []
    for inner_fold, (train_index, val_index) in tqdm(
        enumerate(kf.split(x_train_val, y_train_val)),
        desc="Folds",
        colour="magenta",
        total=num_inner_folds,
        disable=True,
    ):

        x_train, x_val = x_train_val[train_index], x_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]
        if not outlier_flag:
            inner_fold_result = perform_fold_prediction(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
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
            inner_fold_result = perform_only_outlier_detection(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                name=name,
                task_type=task_type,
                output_size=output_size,
                model_args=model_args,
                train_args=train_args,
                experiment_args=experiment_args,
                random_seed=random_seed,
                num_jobs=num_jobs,
            )
        inner_fold_result.update(
            {
                "outer_fold": outer_fold_id,
                "inner_fold": inner_fold,
                "task_num": task_num,
                "experiment_args": experiment_args,
                "model_args": model_args,
                "train_args": train_args,
            }
        )
        inner_fold_results.append(inner_fold_result)
    # return inner_fold_results

    if outer_fold_id is not None:
        output_filename: str = (
            f"task{dataset_id}_dropout_rate{experiment_args['dropout_rate']}_model_precision{experiment_args['alpha']}_num_mcdropout_iterations{experiment_args['mcdropout_num']}_num_layers{experiment_args['num_layers']}_outerfold{outer_fold_id}.pth"
        )
    else:
        output_filename: str = (
            f"task{dataset_id}_dropout_rate{experiment_args['dropout_rate']}_model_precision{experiment_args['alpha']}_num_mcdropout_iterations{experiment_args['mcdropout_num']}_num_layers{experiment_args['num_layers']}.pth"
        )
    # save list of dicts to json
    # TODO: find a better schema. Probably not a good idea to save everything at the end.
    torch.save(inner_fold_results, os.path.join(results_path, output_filename))


def run_dropout_estimation(fn_input):
    dataset, K, y_mean, p, num_mcdropout, num_epochs, batch_sizes, alpha, model_args = (
        fn_input
    )
    # TODO: implement early stopping using val set
    classifier = perform_fold_prediction(
        x_train=dataset["train"][0].astype("float32"),
        y_train=dataset["train"][1].astype("float32"),
        x_val=dataset["val"][0],
        y_val=dataset["val"][1],
        name=(0, 0),
        task_type="regression",
        output_size=1,
        model_args={
            "layer_size": model_args["layer_size"],
            "hidden_activation_type": model_args["hidden_activation"],
        },
        train_args={"num_epochs": num_epochs},
        experiment_args={
            "alpha": alpha,
            "num_layers": model_args['num_layers'],
            "dropout_rate": p,
            "dropout_input": model_args['dropout_input'],
        },
        x_test=dataset["test"][0],
        y_test=dataset["test"][1],
        random_seed=42,
        batch_size=batch_sizes,
    )

    mc_predictions = {
        test_set: np.asarray(
            [
                # NOTE: The predictions are all the same, so I just have to repeat them
                np.repeat(
                    classifier._predict(dataset[test_set][0][[0]]),
                    len(dataset[test_set][0]),
                )
                # classifier._predict(dataset[test_set][0][[0]])
                for _ in tqdm(range(num_mcdropout), disable=True)
            ]
        )
        for test_set in ["val", "test"]
    }
    return (K, y_mean, p, mc_predictions)
