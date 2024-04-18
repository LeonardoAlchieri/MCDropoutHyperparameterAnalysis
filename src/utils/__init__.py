import torch
import random
import numpy


def nanvar(
    tensor: torch.Tensor, dim: int | tuple | None = None, keepdim: bool = False
) -> torch.Tensor:
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


class OutputTypeError(ValueError):
    def __init__(self):
        self.supported_types = ['regression', 'multiclass classification', 'binary classification']
        super().__init__(f"Output type not supported. Supported types are: {', '.join(self.supported_types)}")
        
class NanError(ValueError):
    def __init__(self, message):
        super().__init__(message)
        

def set_seed(random_seed: int) -> None:
    # set reproduction seeds
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # set numpy seed
    numpy.random.seed(random_seed)

    # set any other random seed
    random.seed(random_seed)