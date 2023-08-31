from torch import nn
import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


class sqen_nn(nn.Module):
    """
    神经网络的搭建，包括卷积、池化、摊平、线性
    """

    def __init__(self):
        super(sqen_nn, self).__init__()
        self.moudle1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.moudle1(x)
        return x
