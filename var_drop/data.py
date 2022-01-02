import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

mnist_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
])


def make_mnist_data_loaders(batch_size, download: bool = True):
    train_dataset = MNIST('./data', train=True, download=download, transform=mnist_transforms)
    test_dataset = MNIST('./data', train=False, download=download, transform=mnist_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
