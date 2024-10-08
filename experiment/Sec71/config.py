###
# File: /sgd-influence/experiment/Sec71/config.py
# Created Date: Friday, September 20th 2024
# Author: Zihan
# -----
# Last Modified: Thursday, 26th September 2024 8:38:25 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

# Dictionary to hold configurations for (dataset, network) pairs
DATASET_NETWORK_CONFIG = {
    ("mnist", "logreg"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.003,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # not good
    },
    ("mnist", "dnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.0005,
        "decay": False,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # used
    },
    ("mnist", "cnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.0004,
        "decay": False,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # not tried
    },
    # 20news is not good
    ("20news", "logreg"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
    },
    ("20news", "dnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 200,
        "n_val": 200,
        "n_test": 200,
    },
    ("20news", "cnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.01,
        "decay": True,
        "n_tr": 200,
        "n_val": 200,
        "n_test": 200,
    },
    ("adult", "logreg"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # good
    },
    ("adult", "dnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.3,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # good
    },
    ("adult", "cnn"): {
        "num_epoch": 21,
        "batch_size": 60,
        "lr": 0.02,
        "decay": True,
        "n_tr": 200,
        "n_val": 200,
        "n_test": 200,
        # not tried
    },
    # cifar is not tried
    ("cifar", "logreg"): {
        "num_epoch": 21,
        "batch_size": 60,
        "lr": 0.01,
        "decay": True,
        "n_tr": 200,
        "n_val": 200,
        "n_test": 200,
    },
    ("cifar", "dnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.18,
        "decay": True,
        "n_tr": 256,
        "n_val": 1256,
        "n_test": 200,
    },
    ("cifar", "cnn"): {
        "num_epoch": 50,
        "batch_size": 128,
        "lr": 0.01,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
    },
    ("emnist", "logreg"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.003,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # well, just so so
    },
    ("emnist", "dnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.06,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # good
    },
    ("emnist", "cnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.2,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # not tried
    },
    ("cifar", "cnn_cifar" ): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.01,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
    },
}


def fetch_training_params(dataset: str, network: str):
    """Retrieve the training parameters for a given dataset and network pair."""
    pair_key = (dataset, network)
    if pair_key not in DATASET_NETWORK_CONFIG:
        raise ValueError(
            f"Configuration for dataset {dataset} with network {network} not found."
        )
    return DATASET_NETWORK_CONFIG[pair_key]
