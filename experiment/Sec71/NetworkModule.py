###
# File: /NetworkModule.py
# Created Date: Friday, September 20th 2024
# Author: Zihan
# -----
# Last Modified: Friday, 27th September 2024 9:40:38 pm
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
import logging

NETWORK_REGISTRY = {}


def register_network(key):
    def decorator(cls):
        NETWORK_REGISTRY[key] = cls
        return cls

    return decorator


def get_network(key, input_dim, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    if key not in NETWORK_REGISTRY:
        logger.error(f"Network {key} not found in registry.")
        raise ValueError(f"Network {key} not found in registry.")
    logger.debug(f"Creating network: {key} with input dimension: {input_dim}")
    return NETWORK_REGISTRY[key](input_dim, logger)


class BaseModel(nn.Module):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

    def preprocess_input(self, x):
        raise NotImplementedError

    def forward(self, x):
        self.logger.debug(f"BaseModel forward pass with input shape: {x.shape}")
        x = self.preprocess_input(x)
        return self.model(x)


class NetList(nn.Module):
    def __init__(self, list_of_models, logger=None):
        super(NetList, self).__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.models = nn.ModuleList(list_of_models)
        self.logger.debug(f"Created NetList with {len(list_of_models)} models")

    def forward(self, x, idx=0):
        self.logger.debug(
            f"NetList forward pass with input shape: {x.shape}, using model at index: {idx}"
        )
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
                self.logger.error("Invalid indices for nested NetList access")
                raise ValueError("Invalid indices for nested NetList access")
        return model


@register_network("logreg")
class LogReg(BaseModel):
    def __init__(self, input_dim, logger=None):
        super(LogReg, self).__init__(logger)
        
        # dim of input_dim
        if isinstance(input_dim, (tuple, list, torch.Size)):
            input_dim = np.prod(input_dim)
        
        self.model = nn.Linear(input_dim, 1)
        self.logger.debug(f"Created LogReg model with input dimension: {input_dim}")

    def preprocess_input(self, x):
        # 首先确保 x 是 PyTorch 张量
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # 然后根据维度进行处理
        if x.dim() == 1:
            return x.view(1, -1)
        elif x.dim() == 2:
            return x
        else:
            return x.view(x.size(0), -1)


@register_network("dnn")
class DNN(BaseModel):
    def __init__(self, input_dim, logger=None, m=None):
        super(DNN, self).__init__(logger)
        if m is None:
            m = [8, 8]

        if isinstance(input_dim, (tuple, list, torch.Size)):
            total_input_dim = np.prod(input_dim)
        else:
            total_input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(total_input_dim, m[0]),
            nn.ReLU(),
            nn.Linear(m[0], 1),
        )
        self.logger.debug(
            f"Created DNN model with input dimension: {input_dim}, hidden layers: {m}"
        )

    def preprocess_input(self, x):
        return x.view(x.size(0), -1)

    def param_diff(self, other):
        if not isinstance(other, DNN):
            self.logger.error("Can only compare with another DNN instance")
            raise ValueError("Can only compare with another DNN instance")

        diff = {}
        for (name1, param1), (name2, param2) in zip(
            self.named_parameters(), other.named_parameters()
        ):
            if name1 != name2:
                self.logger.error(f"Parameter names do not match: {name1} vs {name2}")
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

    def print_param_diff(self, other, threshold=1e-6):
        diff = self.param_diff(other)
        for name, diff_tensor in diff.items():
            if torch.any(torch.abs(diff_tensor) > threshold):
                self.logger.debug(f"Difference in {name}:")
                self.logger.debug(diff_tensor)


@register_network("cnn")
class CNN(BaseModel):
    def __init__(self, input_dim, logger=None, m=[32, 64]):
        super(CNN, self).__init__(logger)
        in_channels, height, width = input_dim
        self.m = m
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, self.m[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.m[0], self.m[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(self.m[1] * (height // 4) * (width // 4), 1),
        )
        self.logger.debug(
            f"Created CNN model with input dimension: {input_dim}, channels: {m}"
        )

    def preprocess_input(self, x):
        self.logger.debug(f"CNN input shape before preprocessing: {x.shape}")
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, self.m[0], self.m[1], self.m[2])
        elif x.dim() != 4:
            raise ValueError(f"Unexpected input dimension: {x.shape}")
        self.logger.debug(f"CNN input shape after preprocessing: {x.shape}")
        return x

    def forward(self, x):
        self.logger.debug(f"CNN forward input shape: {x.shape}")

        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Conv2d):
                self.logger.debug(
                    f"Layer {i} ({type(layer).__name__}) expects input channels: {layer.in_channels}"
                )
            elif isinstance(layer, nn.Linear):
                self.logger.debug(
                    f"Layer {i} ({type(layer).__name__}) expects input features: {layer.in_features}"
                )

            x = layer(x)
            self.logger.debug(
                f"Layer {i} ({type(layer).__name__}) output shape: {x.shape}"
            )

        return x


@register_network("cnn_cifar")
class CNN_CIFAR(BaseModel):
    def __init__(self, input_dim, logger=None, m=[32, 32, 64, 64, 128, 128]):
        super(CNN_CIFAR, self).__init__(logger)
        height, width, in_channels = input_dim  # Correct order for CIFAR data
        self.m = m
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=in_channels, out_channels=m[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(m[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[0], out_channels=m[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Layer block 2
            nn.Conv2d(in_channels=m[1], out_channels=m[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(m[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[2], out_channels=m[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Layer block 3
            nn.Conv2d(in_channels=m[3], out_channels=m[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(m[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[4], out_channels=m[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(4 * 4 * m[5], 1),  # Output 1 value for binary classification
        )
        
        self.logger.debug(
            f"Created CNN_CIFAR model with input dimension: {input_dim}, channels: {self.m}"
        )

    def forward(self, x):
        self.logger.debug(f"CNN_CIFAR forward input shape: {x.shape}")
        # Permute the input to (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        self.logger.debug(f"Shape after permute: {x.shape}")
        
        x = self.conv_layer(x)
        self.logger.debug(f"Shape after conv layers: {x.shape}")
        
        x = x.reshape(x.size(0), -1)
        self.logger.debug(f"Shape after flattening: {x.shape}")
        
        x = self.fc_layer(x)
        self.logger.debug(f"Final output shape: {x.shape}")
        
        return x

    def flatten(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layer(x)
        return x.reshape(x.size(0), -1)