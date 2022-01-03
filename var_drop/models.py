import torch
from torch import nn as nn
from torch.nn import functional as F

from .torch_modules import LinearARD


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc1 = LinearARD(in_features=120, out_features=84)
        self.fc2 = LinearARD(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VariationalLeNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        pass
