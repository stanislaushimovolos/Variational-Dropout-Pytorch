import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Callable, List
from torch import nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .utils import PathLike


def optimizer_step(optimizer: Optimizer, loss: torch.Tensor, scaler: torch.cuda.amp.GradScaler = None) -> torch.Tensor:
    optimizer.zero_grad()
    if scaler is not None:
        with torch.cuda.amp.autocast(False):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss


def train_step(inputs: torch.tensor, targets: torch.tensor, model: nn.Module, criterion: Callable,
               optimizer: Optimizer, *, scaler: torch.cuda.amp.GradScaler = None) -> np.ndarray:
    model.train()
    with torch.cuda.amp.autocast(scaler is not None):
        inputs = inputs.to(next(model.parameters()))
        targets = targets.to(next(model.parameters()))
        targets = targets.long()
        loss = criterion(model(inputs), targets)

    optimizer_step(optimizer, loss, scaler=scaler)
    return loss.data.cpu().numpy()


def train(train_data_loader: DataLoader, model: nn.Module, criterion: Callable, n_epochs: int,
          scaler: torch.cuda.amp.grad_scaler, optimizer: Optimizer, checkpoints_path: PathLike,
          validation_callback: Callable = None) -> List:
    checkpoints_path = Path(checkpoints_path)
    validation_score = ""
    bar = tqdm(range(n_epochs), desc='Model train loss', leave=True)

    total_losses = []
    for epoch in range(n_epochs):
        epoch_losses = []
        for batch_idx, batch in enumerate(train_data_loader):
            # enable last FC layer
            images, targets = batch
            # do optimization
            loss = train_step(images, targets, model, criterion, optimizer, scaler=scaler)
            epoch_losses.append(loss.item())

            bar.set_description(desc=f'Train loss {loss}, validation score {validation_score}\n')
            bar.refresh()

        if validation_callback is not None:
            validation_score = validation_callback()

        current_save_path = checkpoints_path / f'model_{epoch}.pth'
        torch.save(model.state_dict(), current_save_path)
        total_losses.append(epoch_losses)
        bar.update(1)

    return total_losses
