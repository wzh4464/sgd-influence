###
# File: /NetworkModule.py
# Created Date: Friday, September 20th 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 21st September 2024 9:24:55 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NETWORK_REGISTRY = {}


def register_network(key):
    def decorator(cls):
        NETWORK_REGISTRY[key] = cls
        return cls

    return decorator


def get_network(key, input_dim):
    if key not in NETWORK_REGISTRY:
        raise ValueError(f"Network {key} not found in registry.")
    return NETWORK_REGISTRY[key](input_dim)


class BaseModel(nn.Module):
    def preprocess_input(self, x):
        raise NotImplementedError

    def forward(self, x):
        x = self.preprocess_input(x)
        return self.model(x)


class NetList(nn.Module):
    def __init__(self, list_of_models):
        super(NetList, self).__init__()
        self.models = nn.ModuleList(list_of_models)

    def forward(self, x, idx=0):
        return self.models[idx](x)

    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.models)

    def get_model(self, *indices):
        model = self
        for idx in indices:
            if isinstance(model, NetList):
                model = model.models[idx]
            else:
                raise ValueError("Invalid indices for nested NetList access")
        return model


@register_network("logreg")
class LogReg(BaseModel):
    def __init__(self, input_dim):
        super(LogReg, self).__init__()
        self.model = nn.Linear(input_dim, 1)

    def preprocess_input(self, x):
        return x.view(x.size(0), -1)


@register_network("dnn")
class DNN(BaseModel):
    def __init__(self, input_dim, m=None):
        super(DNN, self).__init__()
        if m is None:
            m = [8, 8]

        # Calculate the total number of input features
        if isinstance(input_dim, (tuple, list, torch.Size)):
            total_input_dim = np.prod(input_dim)
        else:
            total_input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(total_input_dim, m[0]),
            nn.ReLU(),
            nn.Linear(m[0], 1),
        )

    def preprocess_input(self, x):
        return x.view(x.size(0), -1)

    def param_diff(self, other):
        if not isinstance(other, DNN):
            raise ValueError("Can only compare with another DNN instance")

        diff = {}
        for (name1, param1), (name2, param2) in zip(
            self.named_parameters(), other.named_parameters()
        ):
            if name1 != name2:
                raise ValueError(f"Parameter names do not match: {name1} vs {name2}")
            diff[name1] = param1.data - param2.data
        return diff

    def param_diff_norm(self, other, norm_type=2):
        diff = self.param_diff(other)
        total_norm = sum(
            torch.norm(diff_tensor, p=norm_type).item() ** norm_type
            for diff_tensor in diff.values()
        )
        return total_norm ** (1 / norm_type)

    @staticmethod
    def print_param_diff(diff, threshold=1e-6):
        for name, diff_tensor in diff.items():
            if torch.any(torch.abs(diff_tensor) > threshold):
                print(f"Difference in {name}:")
                print(diff_tensor)
                print()


@register_network("cnn")
class CNN(BaseModel):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        # input_dim is now expected to be a tuple (channels, height, width)
        in_channels, height, width = input_dim

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * (height // 8) * (width // 8), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def preprocess_input(self, x):
        # x is already in the correct shape (batch_size, channels, height, width)
        return x
