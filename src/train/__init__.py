import torch
import torch.nn.functional as F
import torch.optim as optim

# import dataset and dataloader for pytoarch
import torch.utils.data
from accelerate import Accelerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm.auto import tqdm
from gc import collect as pick_up_trash

from src.model import MLP
from src.utils import OutputTypeError


def train(
    x: torch.Tensor,
    y: torch.Tensor,
    num_folds: int,
    num_epochs: int,
    learning_rate: int,
    dropout_rate: float,
    model_precision: float,
    task_num: int,
    batch_size: int,
    length_scale: int,
    learning_rate_epoch_rate: int,
    learning_rate_decay: float,
    random_seed: int,
    model_args: dict,
) -> list[dict]:

    accelerator = Accelerator()

    # Perform k-fold cross validation
    kf = StratifiedKFold(n_splits=num_folds, random_state=random_seed, shuffle=True)
    
    fold = 0
    best_model_infos: list[dict] = []
    for train_index, val_index in tqdm(kf.split(x,y), desc="Folds", colour="red", leave=False, total=num_folds):
        fold += 1

        # FIXME: I don't like that, in each fold, I am ridefining the model and the loss function. This might accidentaly break
        model = MLP(**model_args)
        task_type = model.task_type

        x = torch.tensor(x)
        y = torch.tensor(y)
        # Split the data into training and validation sets
        x_train, x_val = x[train_index].float(), x[val_index].float()
        y_train, y_val = y[train_index].type(torch.long), y[val_index].type(torch.long)

        # Define the train and validation dataloaders
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

        # NOTE: we shuffle in the train set in order to avoid the model
        # seeing the same batches
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        num_samples = len(x_train)
        # reg = (
        #     length_scale**2 * (1 - dropout_rate) / (2.0 * num_samples * model_precision)
        # )
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                            #    weight_decay=reg)

        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )
        # NOTE: I am defining their type  here, since accelerator messes them up
        model: MLP
        optimizer: optim.Adam

        # Since we are using MCDropout, we want the model to always be in training mode
        model.train()

        best_val_mcc: float = (
            -2
        )
        best_model_info = dict()
        # Train the model for the specified number of epochs
        for epoch in tqdm(range(num_epochs), desc="Epochs", colour="green", leave=False):

            train_loss = 0
            if epoch % learning_rate_epoch_rate == 0:
                # lower the learning rate every 3 epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= learning_rate_decay

            for x_train, y_train in tqdm(
                train_dataloader,
                desc="Training Batches",
                colour="yellow",
                leave=False,
                disable=True,
            ):

                optimizer.zero_grad()

                # Forward pass
                y_pred = model(x_train)

                # Calculate the loss
                loss = model.loss_function(y_pred, y_train)

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()

                train_loss += loss.item()

            # Evaluate the model on the validation set
            with torch.no_grad():
                total_val_loss = 0
                
                all_y_val_pred = torch.Tensor()
                all_y_val = torch.Tensor()
                all_y_val_shannon_entropy = torch.Tensor()
                all_y_val_pred_variance = torch.Tensor()
                all_y_val_pred_all_samples = torch.Tensor()
                # validate over the validation set
                for x_val, y_val in tqdm(
                    val_dataloader,
                    desc="Validation Batches",
                    colour="blue",
                    leave=False,
                    disable=True,
                ):
                    # Perform Monte Carlo Dropout during evaluation
                    (
                        y_val_pred_all_samples,
                        y_val_pred_mean,
                        y_val_pred_variance,
                        y_val_pred_shannon_entropy,
                    ) = model.eval_mc_dropout(
                        x=x_val,
                        num_mcdropout_iterations=model_args["num_mcdropout_iterations"],
                        task_type=task_type,
                    )
                    
                    all_y_val_pred = torch.cat(
                        [all_y_val_pred, y_val_pred_mean.cpu()], dim=0
                    )
                    all_y_val = torch.cat([all_y_val, y_val.cpu()], dim=0)
                    all_y_val_shannon_entropy = torch.cat(
                        [all_y_val_shannon_entropy, y_val_pred_shannon_entropy.cpu()], dim=0
                    )
                    all_y_val_pred_variance = torch.cat(
                        [all_y_val_pred_variance, y_val_pred_variance.cpu()], dim=0
                    )
                    # FIXME: check here!
                    all_y_val_pred_all_samples = torch.cat(
                        [all_y_val_pred_all_samples, y_val_pred_all_samples.cpu()], dim=1
                    )

                    # Calculate the validation loss
                    val_loss = model.loss_function(y_val_pred_mean, y_val)

                    total_val_loss += val_loss.item()

                if (
                    task_type == "binary classification"
                ):
                    val_accuracy = accuracy_score(
                        all_y_val.cpu().numpy().astype(int),
                        (all_y_val_pred > model.prediction_threshold).int().cpu().numpy(),
                    )
                    val_f1 = f1_score(
                        all_y_val.cpu().numpy().astype(int),
                        (all_y_val_pred > model.prediction_threshold).int().cpu().numpy(),
                        average=("binary"),
                    )
                    val_mcc = matthews_corrcoef(
                        all_y_val.cpu().numpy().astype(int),
                        (all_y_val_pred > model.prediction_threshold).int().cpu().numpy(),
                    )
                elif task_type == "multiclass classification":
                    all_y_val = all_y_val.cpu().numpy().astype(int)
                    all_y_val_pred = torch.argmax(all_y_val_pred, dim=1).cpu().numpy()
                    val_accuracy = accuracy_score(
                        all_y_val,
                        all_y_val_pred,
                    )
                    val_f1 = f1_score(
                        all_y_val,
                        all_y_val_pred,
                        average=("macro"),
                    )
                    val_mcc = matthews_corrcoef(
                        all_y_val,
                        all_y_val_pred,
                    )
                elif task_type == "regression":
                    raise NotImplementedError(
                        "Regression task type not implemented yet. Please implement the loss function for regression."
                    )
                else:
                    raise OutputTypeError(
                        f"Task type {task_type} not supported"
                    )

                mean_val_shannon_entropy = torch.mean(all_y_val_shannon_entropy)
                mean_val_variance = torch.mean(all_y_val_pred_variance)
                
                # Print the training and validation loss for each epoch
                if val_mcc > best_val_mcc:
                    best_model_info = {
                        # "model_weights": model.state_dict(),
                        "training_info": {
                            "model": model.__class__.__name__,
                            "dataset_openml_id": task_num,
                            "dropout_rate": dropout_rate,
                            "model_precision": model_precision,
                            "num_mcdropout_iterations": model_args[
                                "num_mcdropout_iterations"
                            ],
                            "num_layers": model_args["num_layers"],
                        },
                        # "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": val_accuracy,
                        "val_f1": val_f1,
                        "val_mcc": val_mcc,
                        "val_loss": val_loss.item(),
                        "mean_val_shannon_entropy": mean_val_shannon_entropy.item(),
                        "mean_val_variance": mean_val_variance.item(),
                        "all_y_val_pred_all_samples": all_y_val_pred_all_samples,
                        "train_loss": train_loss,
                        "cross_val_fold": fold,
                        "random_seed": random_seed,
                    }
                    # print(
                    #     f"""
                    #     task{task_num} 
                    #     dropout_rate{dropout_rate} 
                    #     model_precision{model_precision} 
                    #     num_mcdropout_iterations{model_args[
                    #             'num_mcdropout_iterations'
                    #         ]} 
                    #     num_layers{model_args['num_layers']}
                    #     current_learning_rate{param_group['lr']}
                    #     Epoch {epoch+1}/{num_epochs} — Training Loss: {loss.item()}
                    #     — Validation Loss: {val_loss.item()}
                    #     — Mean Validation Uncertainty: {mean_val_uncertainty.item()}
                    #     — Validation Accuracy: {val_accuracy}
                    #     - Validation F1: {val_f1}
                    #     """
                    # )

        best_model_infos.append(best_model_info)
        del model
        pick_up_trash()

    return best_model_infos