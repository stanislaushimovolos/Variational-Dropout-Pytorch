import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F


class ARDMixin:
    def get_kl_loss(self):
        raise NotImplementedError

    def get_dropped_params_number(self):
        raise NotImplementedError

    def get_total_params_number(self):
        raise NotImplementedError


class LinearARD(nn.Module, ARDMixin):
    def __init__(self, in_features: int, out_features: int, threshold: int = 3, eps=1e-10):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(1, out_features))
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.log_sigma2 = nn.Parameter(torch.zeros(out_features, in_features))

        self.init_parameters()

    def init_parameters(self):
        self.bias.data.zero_()
        self.weight.data.normal_(0, 0.02)
        self.log_sigma2.data.fill_(-10)

    def forward(self, x):
        if self.training:
            mean = F.linear(x, self.weight) + self.bias
            std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma2)) + self.eps)
            noise = torch.normal(torch.zeros_like(mean), std)
            return mean + noise * std
        else:
            pruned_weights = self._get_pruned_weights()
            return F.linear(x, pruned_weights) + self.bias

    def _get_clip_mask(self):
        log_alpha = self._get_log_alpha()
        return log_alpha < self.threshold

    def _get_pruned_weights(self):
        mask = self._get_clip_mask()
        return self.weight * mask

    def _get_log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * torch.log(torch.abs(self.weight) + 1e-15)
        log_alpha = torch.clamp(log_alpha, -10, 10)
        return log_alpha

    def get_kl_loss(self):
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self._get_log_alpha()
        kl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha))
        loss = -torch.sum(kl)
        return loss

    def get_dropped_params_number(self):
        mask = self._get_clip_mask()
        return (mask == 0).cpu().numpy().sum()

    def get_total_params_number(self):
        return np.prod(self.weight.shape)


class ConvolutionARD(nn.Module):
    def __init__(self):
        super().__init__()


class ELBOLoss(nn.Module):
    def __init__(self, model, data_loss):
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
        return total_loss


def get_total_params_number(module):
    if isinstance(module, LinearARD) or isinstance(module, ConvolutionARD):
        return module.get_total_params_number()
    elif len(list(module.children())) > 0:
        return sum([get_total_params_number(submodule) for submodule in module.children()])
    return sum(p.numel() for p in module.parameters())


def get_dropped_params_number(module):
    if isinstance(module, LinearARD) or isinstance(module, ConvolutionARD):
        return module.get_dropped_params_number()
    elif len(list(module.children())) > 0:
        return sum([get_dropped_params_number(submodule) for submodule in module.children()])
    else:
        return 0


def calculate_total_sparsity(module):
    return get_dropped_params_number(module) / get_total_params_number(module)
