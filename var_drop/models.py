import torch
from torch import nn as nn
from torch.nn import functional as F
from functools import partial

from .torch_modules import LinearARD, ConvolutionARD


class BaseLeNet(nn.Module):
    def __init__(self, conv_module: nn.Module, linear_module: nn.Module):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(120)
        self.conv1 = conv_module(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = conv_module(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = conv_module(in_channels=16, out_channels=120, kernel_size=5)
        self.fc1 = linear_module(in_features=120, out_features=84)
        self.fc2 = linear_module(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


VanillaLeNet = partial(BaseLeNet, conv_module=nn.Conv2d, linear_module=nn.Linear)
ARDLeNet = partial(BaseLeNet, conv_module=ConvolutionARD, linear_module=LinearARD)
