import torch
import numpy as np
from pathlib import Path
from torch import nn as nn
from typing import Union, Callable

PathLike = Union[str, Path]


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


def inference_step(x: Union[np.ndarray, torch.Tensor], model: nn.Module, use_amp: bool = True,
                   activation: Callable = None) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(use_amp or torch.is_autocast_enabled()):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(next(model.parameters()))
            if activation is None:
                return model(x)
            else:
                return activation(model(x))
