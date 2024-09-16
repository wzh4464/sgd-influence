###
# File: /MyNet.py
# Created Date: September 9th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 16th September 2024 9:13:01 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class NetList(torch.nn.Module):
    def __init__(self, list_of_models):
        super(NetList, self).__init__()
        self.models = torch.nn.ModuleList(list_of_models)

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


class LogReg(nn.Module):
    def __init__(self, input_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        return x
        # x = torch.sigmoid(x)
        # x = torch.cat((x, 1 - x), dim=1)
        # return torch.log(x)


class DNN(nn.Module):
    def __init__(self, input_dim, m=None):
        if m is None:
            m = [8, 8]
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, m[0]),
            # nn.ReLU(),
            # nn.Linear(m[0], m[1]),
            nn.ReLU(),
            nn.Linear(m[1], 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
        # x = torch.sigmoid(x)
        # x = torch.cat((x, 1 - x), dim=1)
        # return torch.log(x)

    def param_diff(self, other):
        """
        Calculate the difference between this model's parameters and another model's parameters.

        :param other: Another DNN model to compare with
        :return: A dictionary containing the differences of each parameter
        """
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
        """
        Calculate the norm of the difference between this model's parameters and another model's parameters.

        :param other: Another DNN model to compare with
        :param norm_type: Type of norm to use (default is L2 norm)
        :return: The norm of the parameter differences
        """
        diff = self.param_diff(other)
        total_norm = sum(
            torch.norm(diff_tensor, p=norm_type).item() ** norm_type
            for diff_tensor in diff.values()
        )
        return total_norm ** (1 / norm_type)

    @staticmethod
    def print_param_diff(diff, threshold=1e-6):
        """
        Print the parameter differences, showing only differences above a certain threshold.

        :param diff: Dictionary of parameter differences
        :param threshold: Minimum absolute difference to display
        """
        for name, diff_tensor in diff.items():
            if torch.any(torch.abs(diff_tensor) > threshold):
                print(f"Difference in {name}:")
                print(diff_tensor)
                print()


class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
