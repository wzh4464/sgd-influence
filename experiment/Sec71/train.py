###
# File: experiment/Sec71/train.py
# Created Date: September 9th 2024
# Author: Zihan
# -----
# Last Modified: Thursday, 26th September 2024 8:27:05 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import argparse
import copy
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
import traceback
import pandas as pd
from logging_utils import setup_logging
import logging
from NetworkModule import get_network, NETWORK_REGISTRY, NetList
import warnings

# Assuming these imports are from local files
from DataModule import DATA_MODULE_REGISTRY

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

file_abspath = os.path.abspath(__file__)
current_dir = os.path.dirname(file_abspath)  # 获取当前脚本所在的目录路径


from DataModule import fetch_data_module
from config import fetch_training_params

import logging

# import functools

# def monitor_level_change(func):
#     @functools.wraps(func)
#     def wrapper(self, level):
#         old_level = self.level
#         result = func(self, level)
#         if old_level != self.level:
#             print(f"Logger '{self.name}' level changed from {old_level} to {self.level}")
#             # 这里可以设置一个断点
#         return result
#     return wrapper

# # 应用猴子补丁
# logging.Logger.setLevel = monitor_level_change(logging.Logger.setLevel)


def initialize_data_and_params(
    key: str, model_type: str, csv_path: str, logger=None, seed: int = 0
) -> Tuple[Any, Dict[str, int], Dict[str, Any]]:
    """Initialize the data module and fetch training parameters for a dataset and model."""
    module = fetch_data_module(key, data_dir=csv_path, logger=logger, seed=seed)

    # Fetch the training parameters from the config file based on dataset and network
    config = fetch_training_params(key, model_type)

    # Use config values if available, otherwise use defaults
    training_params = {
        "num_epoch": config.get("num_epoch", 21),
        "batch_size": config.get("batch_size", 60),
        "lr": config.get("lr", 0.01),
        "decay": config.get("decay", True),
    }

    data_sizes = {
        "n_tr": config.get("n_tr", 200),
        "n_val": config.get("n_val", 200),
        "n_test": config.get("n_test", 200),
    }

    module.append_one = False
    return module, data_sizes, training_params


def get_model(model_type: str, input_dim: int, device: str, logger=None):
    return get_network(model_type, input_dim, logger).to(device)


def load_data(
    key: str,
    model_type: str,
    seed: int,
    csv_path: str,
    custom_n_tr: int = None,
    custom_n_val: int = None,
    custom_n_test: int = None,
    custom_num_epoch: int = None,
    custom_batch_size: int = None,
    custom_lr: float = None,
    relabel_percentage: float = None,
    device: str = "cpu",
    logger=None,
):
    # Fetch data and settings
    module, data_sizes, training_params = initialize_data_and_params(
        key, model_type, csv_path, logger, seed
    )

    # Override default values if custom values are provided
    if custom_n_tr:
        data_sizes["n_tr"] = custom_n_tr
    if custom_n_val:
        data_sizes["n_val"] = custom_n_val
    if custom_n_test:
        data_sizes["n_test"] = custom_n_test
    if custom_num_epoch:
        training_params["num_epoch"] = custom_num_epoch
    if custom_batch_size:
        training_params["batch_size"] = custom_batch_size
    if custom_lr:
        training_params["lr"] = custom_lr

    z_tr, z_val, _ = module.fetch(
        data_sizes["n_tr"], data_sizes["n_val"], data_sizes["n_test"], seed
    )
    (x_tr, y_tr), (x_val, y_val) = z_tr, z_val

    # Relabel a percentage of training data if specified
    relabeled_indices = None
    if relabel_percentage is not None and relabel_percentage > 0:
        num_to_relabel = int(data_sizes["n_tr"] * relabel_percentage / 100)
        relabeled_indices = np.random.choice(
            data_sizes["n_tr"], num_to_relabel, replace=False
        )
        y_tr[relabeled_indices] = 1 - y_tr[relabeled_indices]
        logger.info(
            f"Relabeled {num_to_relabel} samples ({relabel_percentage}% of training data)"
        )

    # Convert to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(y_tr).to(torch.float32).unsqueeze(1).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(y_val).to(torch.float32).unsqueeze(1).to(device)

    return x_tr, y_tr, x_val, y_val, data_sizes, training_params, relabeled_indices


def save_at_initial(
    model,
    net_func,
    list_of_sgd_models,
    list_of_counterfactual_models,
    n,
    compute_counterfactual,
    logger,
):
    m = net_func()
    m.load_state_dict(copy.deepcopy(model.state_dict()))
    m.to("cpu")
    if n == -1:
        list_of_sgd_models.append(m)
        logger.debug(
            f"Saved initial SGD model. Total models: {len(list_of_sgd_models)}"
        )

    if compute_counterfactual and n >= 0:
        list_of_counterfactual_models[n] = NetList([m])
        logger.debug(f"Saved initial model for counterfactual sample {n}")


def save_after_epoch(
    model,
    net_func,
    list_of_counterfactual_models,
    n,
    epoch,
    compute_counterfactual,
    logger,
):
    if compute_counterfactual and n >= 0:
        m = net_func()
        m.load_state_dict(copy.deepcopy(model.state_dict()))
        m.to("cpu")
        list_of_counterfactual_models[n].models.append(m)
        logger.debug(f"Saved model for counterfactual sample {n}, epoch {epoch+1}")


def save_after_step(model, net_func, list_of_sgd_models, n, total_step, logger):
    if n == -1:
        m = net_func()
        m.load_state_dict(copy.deepcopy(model.state_dict()))
        m.to("cpu")
        list_of_sgd_models.append(m)
        logger.debug(
            f"Saved SGD model at step {total_step+1}. Total models: {len(list_of_sgd_models)}"
        )


def train_and_save(
    key: str,
    model_type: str,
    seed: int = 0,
    gpu: int = 0,
    csv_path: str = None,
    custom_n_tr: int = None,
    custom_n_val: int = None,
    custom_n_test: int = None,
    custom_num_epoch: int = None,
    custom_batch_size: int = None,
    custom_lr: float = None,
    relabel_percentage: float = None,
    compute_counterfactual: bool = True,
    logger=None,
    save_dir: str = None,
) -> Dict[str, Any]:
    if csv_path is None:
        csv_path = os.path.join(current_dir, "data")

    if save_dir is None:
        save_dir = f"{key}_{model_type}"

    dn = os.path.join(current_dir, save_dir)
    fn = os.path.join(dn, f"sgd{seed:03d}.dat")
    os.makedirs(dn, exist_ok=True)

    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS")
    else:
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = f"cuda:{gpu}"
        else:
            # CUDA is not available, fall back to CPU and throw a warning
            warnings.warn("CUDA is not available, using CPU instead.", UserWarning)
            device = "cpu"

    # Load data
    x_tr, y_tr, x_val, y_val, data_sizes, training_params, relabeled_indices = (
        load_data(
            key,
            model_type,
            seed,
            csv_path,
            custom_n_tr,
            custom_n_val,
            custom_n_test,
            custom_num_epoch,
            custom_batch_size,
            custom_lr,
            relabel_percentage,
            device,
            logger,
        )
    )

    logger.debug(
        f"Dataset {key} loaded with {data_sizes['n_tr']} training samples, {data_sizes['n_val']} validation samples"
    )

    # Model selection and hyperparameter tuning
    if model_type == "logreg":
        model = LogisticRegressionCV(random_state=seed, fit_intercept=False, cv=5)
        # reshape x_tr to 2D
        x_tr = x_tr.view(data_sizes["n_tr"], -1)
        y_tr = y_tr.view(data_sizes["n_tr"])

        x_tr_npclone = x_tr.clone().detach().cpu().numpy()
        y_tr_npclone = y_tr.clone().detach().cpu().numpy()

        model.fit(x_tr_npclone, y_tr_npclone)
        alpha = 1 / (model.C_[0] * data_sizes["n_tr"])

        y_tr = y_tr.float().view(-1, 1)
    else:
        alpha = 0.001  # You might want to tune this for DNN/CNN

    logger.debug(f"Model {model_type} initialized with alpha={alpha}")

    # Get input dimension for the model
    input_dim = x_tr.shape[1:]

    # Training setup
    net_func = lambda: get_model(model_type, input_dim, device, logger)
    num_steps = int(np.ceil(data_sizes["n_tr"] / training_params["batch_size"]))
    list_of_counterfactual_models = (
        [NetList([]) for _ in range(data_sizes["n_tr"])]
        if compute_counterfactual
        else None
    )

    # Initialize metrics and models
    list_of_sgd_models = []
    list_of_counterfactual_models = (
        [NetList([]) for _ in range(data_sizes["n_tr"])]
        if compute_counterfactual
        else None
    )

    def train_single_model(n, loss_fn=nn.BCEWithLogitsLoss()):
        torch.manual_seed(seed)
        model = net_func()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_params["lr"],
            momentum=0.0,
        )
        lr_n = training_params["lr"]
        skip = [n]
        info = []
        main_losses = []
        test_accuracies = []
        train_losses = []
        total_step = 0

        # Initial evaluation
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
            main_losses.append(val_loss)
            train_losses.append(np.nan)
            test_pred = (model(x_val) > 0).float()
            test_acc = (test_pred == y_val).float().mean().item()
            test_accuracies.append(test_acc)
            logger.debug(
                f"Initial Validation Loss for n={n}: {val_loss:.4f}, Initial Test Accuracy: {test_acc:.4f}"
            )
        
        save_at_initial(model, net_func, list_of_sgd_models if n == -1 else None, list_of_counterfactual_models, n, compute_counterfactual, logger)

        for epoch in range(training_params["num_epoch"]):
            epoch_loss = 0.0
            np.random.seed(epoch)
            idx_list = np.array_split(
                np.random.permutation(data_sizes["n_tr"]), num_steps
            )
            for i in range(num_steps):
                info.append({"idx": idx_list[i], "lr": lr_n})

                # SGD optimization
                idx = idx_list[i]
                b = idx.size
                idx = np.setdiff1d(idx, skip)
                z = model(x_tr[idx])
                loss = loss_fn(z, y_tr[idx])
                epoch_loss += loss.item()

                # Add regularization
                for p in model.parameters():
                    loss += 0.5 * alpha * (p * p).sum()
                optimizer.zero_grad()
                loss.backward()
                for p in model.parameters():
                    p.grad.data *= idx.size / b
                optimizer.step()

                # Learning rate decay
                if training_params["decay"]:
                    lr_n *= np.sqrt((total_step + 1) / (total_step + 2))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_n

                if n == -1:
                    save_after_step(model, net_func, list_of_sgd_models, n, total_step, logger)
                total_step += 1

                # Clean up
                del z, loss
                torch.cuda.empty_cache()

            # End of epoch evaluation
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val).item()
                main_losses.append(val_loss)
                test_pred = (model(x_val) > 0).float()
                test_acc = (test_pred == y_val).float().mean().item()
                test_accuracies.append(test_acc)
                train_losses.append(epoch_loss / num_steps)
                logger.debug(
                    f"n={n}, Epoch {epoch+1}/{training_params['num_epoch']}, "
                    f"Validation Loss: {val_loss:.4f}, Test Accuracy: {test_acc:.4f}, "
                    f"Average Training Loss: {epoch_loss/num_steps:.4f}"
                )

            if n != -1:
                save_after_epoch(model, net_func, list_of_counterfactual_models, n, epoch, compute_counterfactual, logger)

            torch.cuda.empty_cache()

        return model, info, main_losses, test_accuracies, train_losses

    # Main training loop
    logger.debug(f"Starting training for {training_params['num_epoch']} epochs")
    for n in range(-1, data_sizes["n_tr"] if compute_counterfactual else 0):
        logger.info(f"Training model {n+1}/{data_sizes['n_tr']}")
        model, info, main_losses, test_accuracies, train_losses = train_single_model(n)

        if n == -1:
            sgd_main_losses = main_losses
            sgd_test_accuracies = test_accuracies
            sgd_train_losses = train_losses
            sgd_info = info
        elif compute_counterfactual:
            logger.debug(
                f"Number of counterfactual models saved for sample {n}: {len(list_of_counterfactual_models[n].models)}"
            )

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Verify the number of saved models
    expected_sgd_models = training_params["num_epoch"] * num_steps + 1  # Initial model + models for each step
    if len(list_of_sgd_models) != expected_sgd_models:
        logger.warning(f"Unexpected number of SGD models. Expected {expected_sgd_models}, got {len(list_of_sgd_models)}")
    else:
        logger.info(f"Correct number of SGD models saved: {len(list_of_sgd_models)}")

    if compute_counterfactual:
        for i, models in enumerate(list_of_counterfactual_models):
            expected_models = training_params["num_epoch"] + 1  # Initial model + models for each epoch
            if len(models.models) != expected_models:
                logger.warning(
                    f"Unexpected number of counterfactual models for sample {i}. Expected {expected_models}, got {len(models.models)}"
                )
            else:
                logger.debug(
                    f"Correct number of counterfactual models saved for sample {i}: {len(models.models)}"
                )

    # Save data
    data_to_save = {
        "models": NetList(list_of_sgd_models),
        "info": sgd_info,
        "counterfactual": list_of_counterfactual_models,
        "alpha": alpha,
        "main_losses": sgd_main_losses,
        "test_accuracies": sgd_test_accuracies,
        "train_losses": sgd_train_losses,
        "seed": seed,
        "n_tr": data_sizes["n_tr"],
        "n_val": data_sizes["n_val"],
        "n_test": data_sizes["n_test"],
        "num_epoch": training_params["num_epoch"],
        "batch_size": training_params["batch_size"],
        "lr": training_params["lr"],
        "decay": training_params["decay"],
        "relabeled_indices": relabeled_indices,
    }

    torch.save(data_to_save, fn)

    # Save metrics to CSV
    csv_fn = os.path.join(dn, f"metrics_{seed:03d}.csv")
    pd.DataFrame(
        {
            "epoch": range(len(sgd_main_losses)),
            "main_loss": sgd_main_losses,
            "test_accuracy": sgd_test_accuracies,
            "train_loss": sgd_train_losses,
        }
    ).to_csv(csv_fn, index=False)

    logger.debug(f"Training completed. Results saved to {fn} and {csv_fn}")

    return data_to_save


def _validate_arguments(logger, args):
    logger.debug("Starting the training process")
    logger.debug(f"Arguments: {args}")

    if args.target not in DATA_MODULE_REGISTRY:
        raise ValueError(
            f"Invalid target data: {args.target}. Available targets: {', '.join(DATA_MODULE_REGISTRY.keys())}"
        )

    if args.model not in NETWORK_REGISTRY:
        raise ValueError(
            f"Invalid model type: {args.model}. Available models: {', '.join(NETWORK_REGISTRY.keys())}"
        )

    # Fetch default configuration for this dataset-model pair
    default_config = fetch_training_params(args.target, args.model)

    if args.seed >= 0:
        _run_training(args, default_config, logger)
    else:
        for seed in range(100):
            args.seed = seed
            _run_training(args, default_config, logger)

    logger.debug("Training process completed successfully")


def _run_training(args, default_config, logger):
    train_and_save(
        args.target,
        args.model,
        args.seed,
        args.gpu,
        custom_n_tr=args.n_tr,
        custom_n_val=args.n_val,
        custom_n_test=args.n_test,
        custom_num_epoch=args.num_epoch,
        custom_batch_size=args.batch_size,
        custom_lr=args.lr,
        relabel_percentage=args.relabel,
        compute_counterfactual=args.compute_counterfactual,
        logger=logger,
        save_dir=args.save_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Models & Save")
    parser.add_argument("--target", default="adult", type=str, help="target data")
    parser.add_argument("--model", default="logreg", type=str, help="model type")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="gpu index")
    parser.add_argument("--n_tr", type=int, help="number of training samples")
    parser.add_argument("--n_val", type=int, help="number of validation samples")
    parser.add_argument("--n_test", type=int, help="number of test samples")
    parser.add_argument("--num_epoch", type=int, help="number of epochs")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument(
        "--save_dir", type=str, help="directory to save models and results"
    )
    parser.add_argument(
        "--no-loo",
        action="store_false",
        dest="compute_counterfactual",
        help="Disable the computation of counterfactual models (leave-one-out).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--relabel", type=float, help="percentage of training data to relabel"
    )

    parser.set_defaults(compute_counterfactual=True)

    args = parser.parse_args()

    # 设置保存目录
    if args.save_dir is None:
        args.save_dir = f"{args.target}_{args.model}"

    # 创建一个 logger 实例
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(
        f"{args.target}_{args.model}",
        args.seed,
        os.path.join(current_dir, args.save_dir),
        level=log_level,
    )

    try:
        _validate_arguments(logger, args)
    except ValueError as e:
        logger.error(f"Invalid argument: {str(e)}")
    except Exception as e:
        logger.error(f"An error occurred during the training process: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
