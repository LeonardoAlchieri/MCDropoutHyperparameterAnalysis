import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from src.utils import nanvar, OutputTypeError, NanError


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_layer_size: int,
        prediction_threshold: float | None = 0.5,
        hidden_activation_type: str = "relu",
        output_type: str = "regression",
        output_size: int = 1,
        num_mcdropout_iterations: int = 2,
        dropout_rate: float = 0.05,
    ) -> None:
        super(MLP, self).__init__()

        self._set_hidden_activation(hidden_activation_type)
        self._set_output_activation(output_type)
        self._set_mcdropout_iterations(num_mcdropout_iterations)
        self._set_dropout_rate(dropout_rate)
        self._set_loss_function()

        # Define the main layers in the network. We use a simple structure
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_layer_size))  # First layer
        self.layers.append(nn.BatchNorm1d(hidden_layer_size))
        self.layers.append(self.activation)
        self.layers.append(nn.Dropout(dropout_rate))  # Dropout layer
        for _ in range(num_layers - 1):
            self.layers.append(
                nn.Linear(hidden_layer_size, hidden_layer_size)
            )  # Hidden layers
            self.layers.append(nn.BatchNorm1d(hidden_layer_size))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(dropout_rate))  # Dropout layer

        self.output_layer = nn.Linear(hidden_layer_size, output_size)

        if self.task_type == "binary classification":
            self.prediction_threshold = prediction_threshold
        else:
            self.prediction_threshold = None

    def _set_hidden_activation(self, hidden_activation_type: str) -> None:
        if hidden_activation_type == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Activation type {hidden_activation_type} not supported. Supported types are: relu."
            )

    def _set_output_activation(self, output_type: str) -> None:
        if output_type == "regression":
            self.output_activation = nn.Identity()
            self.task_type = output_type
        elif output_type == "binary classification":
            self.output_activation = nn.Sigmoid()
            self.task_type = "binary classification"
        elif output_type == "multiclass classification":
            self.output_activation = nn.Softmax(dim=1)
            self.task_type = output_type
        elif output_type == "multilabel classification":
            self.output_activation = nn.Sigmoid()
            self.task_type = "multilabel classification"
        else:
            raise ValueError(
                f"Output type {output_type} not supported. Supported types are: regression, binary classification, multiclass classification, multilabel classification"
            )

    def _set_mcdropout_iterations(self, num_mcdropout_iterations: int) -> None:
        if num_mcdropout_iterations > 1:
            self.num_mcdropout_iterations = num_mcdropout_iterations
        else:
            raise (
                ValueError(
                    f"num_mcdropout_iterations must be greater than 1. Found {num_mcdropout_iterations}."
                )
            )

    def _set_dropout_rate(self, dropout_rate: float) -> None:
        if isinstance(dropout_rate, float):
            if dropout_rate > 0 and dropout_rate < 1:
                self.dropout_rate = dropout_rate
            else:
                raise (
                    ValueError(
                        f"dropout_rate must be between 0 and 1. Found {dropout_rate}."
                    )
                )
        elif isinstance(dropout_rate, int):
            if dropout_rate == 0:
                self.dropout_rate = float(dropout_rate)
            else:
                raise (
                    ValueError(
                        f"if dropout_rate is int, must be 0. Found {dropout_rate}."
                    )
                )
        else:
            raise TypeError(
                f"dropout_rate must be a float or 0. Found {type(dropout_rate)} with value {dropout_rate}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any():
            raise NanError(f"Found NaN values in the input tensor {x}")
        for layer in self.layers:
            x = layer(x)

        x = self.output_layer(x)
        x = self.output_activation(x)

        if torch.isnan(x).any():
            raise NanError(f"Found NaN values in the output tensor {x}")
        return x

    def eval_mc_dropout(
        self,
        x: torch.Tensor,
        num_mcdropout_iterations: int,
        task_type: str,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """This method runs the MCDropout at validation time. It returns a list of predictions

        Parameters
        ----------
        x : torch.Tensor
            input tensor
        num_mcdropout_iterations : int
            number of iterations to perform the MCDropout
        task_type : str
            type of task to perform. It can be either 'binary classification', 'multiclass classification' or 'regression'

        Returns
        -------
        tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]
            the output is composed as follow: a list of tensors, containing each predictions
            for the num_mcdropout_iterations performed to obtain a Montecarlo sample; the mean
            of the sample and the computed uncertainty (variance if regression, entropy if classification)
        """

        # NOTE: I have to save all of the samples, as well as calculate
        # the following statistics: mean, variance, entropy
        dropout_sample = [self.forward(x) for _ in range(num_mcdropout_iterations)]
        # we put the samples along a new dimension
        dropout_sample = torch.stack(dropout_sample)
        mean = torch.nanmean(dropout_sample, dim=0)

        if task_type == "binary classification":
            shannon_entropy = -torch.nansum(
                dropout_sample * torch.log(dropout_sample), dim=0
            )
            variance = nanvar(dropout_sample, dim=0)
        elif task_type == "multiclass classification":
            mean_argmax = torch.argmax(mean, dim=1)
            mean_argmax = mean_argmax.reshape(1, mean_argmax.shape[0], 1).repeat(dropout_sample.shape[0], 1, 1)
            # array shape: (dropout_samples, batch_size, num_classes)
            shannon_entropy = -torch.nansum(
                dropout_sample.gather(2, mean_argmax)
                * torch.log(dropout_sample.gather(2, mean_argmax)),
                dim=0,
            )
            variance = nanvar(dropout_sample.gather(2, mean_argmax), dim=0)
        elif task_type == "regression":
            shannon_entropy = variance
        else:
            raise OutputTypeError(f"Found {task_type}.")

        return dropout_sample, mean, variance.reshape(-1,), shannon_entropy.reshape(-1,)

    def _set_loss_function(self) -> callable:
        if self.task_type == "binary classification":
            self.loss_function = F.binary_cross_entropy
        elif self.task_type == "multiclass classification":
            self.loss_function = F.cross_entropy
        elif self.task_type == "regression":
            raise NotImplementedError(
                "Regression task type not implemented yet. Please implement the loss function for regression."
            )
        else:
            raise OutputTypeError(f"Found {self.task_type}.")
