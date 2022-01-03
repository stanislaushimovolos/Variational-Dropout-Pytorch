import torch
import numpy as np
from typing import Union
from torch import nn as nn
from torch.nn import functional as F

from .utils import to_numpy

TorchLike = Union[torch.Tensor, nn.Parameter]


class ARDMixin:
    def __init__(self, weight: TorchLike, log_sigma2: nn.Parameter, threshold: float, eps: float):
        self.eps = eps
        self.threshold = threshold
        self.weight = weight
        self.log_sigma2 = log_sigma2

    def get_dropped_params_number(self):
        mask = self._get_clip_mask()
        return (mask == 0).cpu().numpy().sum()

    def get_total_params_number(self):
        return np.prod(self.weight.shape)

    def _get_clip_mask(self):
        log_alpha = self._get_log_alpha()
        return log_alpha < self.threshold

    def _get_pruned_weights(self):
        mask = self._get_clip_mask()
        return self.weight * mask

    def _get_log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * torch.log(torch.abs(self.weight) + self.eps)
        log_alpha = torch.clamp(log_alpha, -10, 10)
        return log_alpha

    def get_kl_loss(self):
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self._get_log_alpha()
        kl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha))
        loss = -torch.sum(kl)
        return loss


class LinearARD(nn.Module, ARDMixin):
    def __init__(self, in_features: int, out_features: int, threshold: int = 3, eps=1e-10):
        nn.Module.__init__(self)
        self.eps = eps
        self.threshold = threshold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(1, out_features))
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.log_sigma2 = nn.Parameter(torch.zeros(out_features, in_features))

        self.init_parameters()
        ARDMixin.__init__(self, self.weight, self.log_sigma2, self.threshold, self.eps)

    def init_parameters(self):
        self.bias.data.zero_()
        self.weight.data.normal_(0, 0.02)
        self.log_sigma2.data.fill_(-10)

    def forward(self, x: torch.Tensor):
        if self.training:
            mean = F.linear(x, self.weight) + self.bias
            std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma2)) + self.eps)
            noise = torch.normal(torch.zeros_like(mean), std)
            return mean + noise * std
        else:
            pruned_weights = self._get_pruned_weights()
            return F.linear(x, pruned_weights) + self.bias


class ConvolutionARD(nn.Conv2d, ARDMixin):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, threshold: int = 3, eps=1e-10):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias=False)
        self.eps = eps
        self.bias = None
        self.threshold = threshold
        self.log_sigma2 = nn.Parameter(-10 * torch.ones_like(self.weight))
        ARDMixin.__init__(self, self.weight, self.log_sigma2, self.threshold, self.eps)

    def forward(self, x: torch.Tensor):
        if self.training:
            mean = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            sigma2 = torch.exp(self.log_sigma2)
            var = F.conv2d(x ** 2, sigma2, self.bias, self.stride, self.padding, self.dilation, self.groups)
            std = torch.sqrt(self.eps + var)
            noise = torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
            return mean + noise * std
        else:
            pruned_weights = self._get_pruned_weights()
            return F.conv2d(x, pruned_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ELBOLoss(nn.Module):
    def __init__(self, model: nn.Module, data_loss: nn.Module):
        super().__init__()
        self.model = model
        self.data_loss = data_loss

    def forward(self, x, target, kl_weight=0.0001):
        def get_kl_term(module):
            if isinstance(module, LinearARD) or isinstance(module, ConvolutionARD):
                return module.get_kl_loss()
            elif len(list(module.children())) > 0:
                return sum([get_kl_term(submodule) for submodule in module.children()])
            else:
                return 0

        kl_term = kl_weight * get_kl_term(self.model)
        data_term = self.data_loss(x, target)
        total_loss = kl_term + data_term
        return total_loss, {
            'loss': float(to_numpy(total_loss)),
            'kl_term': float(to_numpy(kl_term)),
            'data_term': float(to_numpy(data_term)),
            'sparsity': calculate_total_sparsity(self.model)
        }


def get_total_params_number(module: nn.Module):
    if isinstance(module, LinearARD) or isinstance(module, ConvolutionARD):
        return module.get_total_params_number()
    elif len(list(module.children())) > 0:
        return sum([get_total_params_number(submodule) for submodule in module.children()])
    return sum(p.numel() for p in module.parameters())


def get_dropped_params_number(module: nn.Module):
    if isinstance(module, LinearARD) or isinstance(module, ConvolutionARD):
        return module.get_dropped_params_number()
    elif len(list(module.children())) > 0:
        return sum([get_dropped_params_number(submodule) for submodule in module.children()])
    else:
        return 0


def calculate_total_sparsity(module: nn.Module):
    return get_dropped_params_number(module) / get_total_params_number(module)
