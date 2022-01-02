import torch
import numpy as np

from tqdm import tqdm
from torch import nn as nn
from torch.optim import Optimizer
from typing import Callable, Union
from pathlib import Path
from torch.utils.data import DataLoader

PathLike = Union[str, Path]


def optimizer_step(optimizer: Optimizer, loss: torch.Tensor,
                   scaler: torch.cuda.amp.GradScaler = None, ) -> torch.Tensor:
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


def train(train_data_loader: DataLoader, model: nn.Module, criterion: Callable, n_iterations: int,
          checkpoints_freq: int, scaler: torch.cuda.amp.grad_scaler, optimizer: Optimizer,
          lr_scheduler: torch.optim.lr_scheduler, checkpoints_path: PathLike, validation_callback: Callable = None):
    checkpoints_path = Path(checkpoints_path)

    validation_score = ""
    bar = tqdm(range(n_iterations), desc='Model train loss', leave=True)
    cur_iteration = 0

    while True:
        epoch_losses = []
        for batch_idx, batch in enumerate(train_data_loader):
            # enable last FC layer
            images, targets = batch
            # do optimization
            loss = train_step(images, targets, model, criterion, optimizer, scaler=scaler)
            epoch_losses.append(loss)

            current_lr = lr_scheduler.get_last_lr()
            bar.set_description(
                desc=f'Train loss {loss}, iter {cur_iteration}, lr {current_lr} validation score {validation_score}\n')
            bar.refresh()
            lr_scheduler.step()
            if cur_iteration % checkpoints_freq == 0:
                if validation_callback is not None:
                    validation_score = validation_callback()

                current_save_path = checkpoints_path / f'checkpoint_{cur_iteration}.pth'
                torch.save(model.state_dict(), current_save_path)

            cur_iteration += 1
            if cur_iteration >= n_iterations:
                break
