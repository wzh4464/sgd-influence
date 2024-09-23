###
# File: experiment/Sec71/train.py
# Created Date: September 9th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 23rd September 2024 10:25:13 am
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
import functools

def monitor_level_change(func):
    @functools.wraps(func)
    def wrapper(self, level):
        old_level = self.level
        result = func(self, level)
        if old_level != self.level:
            print(f"Logger '{self.name}' level changed from {old_level} to {self.level}")
            # 这里可以设置一个断点
        return result
    return wrapper

# 应用猴子补丁
logging.Logger.setLevel = monitor_level_change(logging.Logger.setLevel)

def initialize_data_and_params(
    key: str, model_type: str, csv_path: str, logger=None, seed: int = 0
) -> Tuple[Any, Dict[str, int], Dict[str, Any]]:
    """Initialize the data module and fetch training parameters for a dataset and model."""
    module = fetch_data_module(key, data_dir=csv_path,logger=logger, seed=seed)

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

    # Convert to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(y_tr).to(torch.float32).unsqueeze(1).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(y_val).to(torch.float32).unsqueeze(1).to(device)

    return x_tr, y_tr, x_val, y_val, data_sizes, training_params


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
    x_tr, y_tr, x_val, y_val, data_sizes, training_params = load_data(
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
        device,
        logger
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
    input_dim = x_tr.shape[1:] # if model_type not in ["cnn_cifar"] else (3, x_tr.shape[1], x_tr.shape[2]) 

    # Training setup
    net_func = lambda: get_model(model_type, input_dim, device, logger)
    num_steps = int(np.ceil(data_sizes["n_tr"] / training_params["batch_size"]))
    list_of_sgd_models = []
    list_of_counterfactual_models = (
        [NetList([]) for _ in range(data_sizes["n_tr"])]
        if compute_counterfactual
        else None
    )
    main_losses = []
    test_accuracies = []
    train_losses = [np.nan]

    logger.debug(f"Starting training for {training_params['num_epoch']} epochs")

    # Training loop
    for n in range(-1, data_sizes["n_tr"] if compute_counterfactual else 0):
        logger.info(f"Training model {n+1}/{data_sizes['n_tr']}")
        torch.manual_seed(seed)
        model = net_func()
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_params["lr"],
            momentum=0.0,
        )

        lr_n = training_params["lr"]
        skip = [n]
        info = []
        c = 0

        for epoch in range(training_params["num_epoch"]):
            epoch_loss = 0.0
            np.random.seed(epoch)
            idx_list = np.array_split(
                np.random.permutation(data_sizes["n_tr"]), num_steps
            )
            for i in range(num_steps):
                info.append({"idx": idx_list[i], "lr": lr_n})
                c += 1

                # Save models and losses
                m = net_func()
                m.load_state_dict(copy.deepcopy(model.state_dict()))
                if n < 0:
                    m.to("cpu")
                    list_of_sgd_models.append(m)
                    if (
                        c % num_steps == 0
                        or c == num_steps * training_params["num_epoch"]
                    ):
                        with torch.no_grad():
                            val_loss = loss_fn(model(x_val), y_val).item()
                            main_losses.append(val_loss)
                            test_pred = (model(x_val) > 0).float()
                            test_acc = (test_pred == y_val).float().mean().item()
                            test_accuracies.append(test_acc)
                            logger.debug(
                                f"Epoch {epoch+1}/{training_params['num_epoch']}, "
                                f"Validation Loss: {val_loss:.4f}, Test Accuracy: {test_acc:.4f}"
                            )
                elif compute_counterfactual:
                    if (
                        c % num_steps == 0
                        or c == num_steps * training_params["num_epoch"]
                    ):
                        m.to("cpu")
                        list_of_counterfactual_models[n].models.append(m)

                # SGD optimization
                idx = idx_list[i]
                b = idx.size
                idx = np.setdiff1d(idx, skip)
                
                logger.debug(f"Shape of x_tr[idx]: {x_tr[idx].shape}")
                logger.debug(f"Type of model: {type(model)}")

                z = model(x_tr[idx])
                loss = loss_fn(z, y_tr[idx])

                if (
                    c % num_steps == 0 or c == num_steps * training_params["num_epoch"]
                ) and n < 0:
                    train_losses.append(loss.item())

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
                    lr_n *= np.sqrt(c / (c + 1))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_n

                # Clean up
                del z, loss
                torch.cuda.empty_cache()

            # End of epoch logging
            logger.debug(
                f"Epoch {epoch+1}/{training_params['num_epoch']}, "
                f"Average Training Loss: {epoch_loss/num_steps:.4f}"
            )
            torch.cuda.empty_cache()

        # Save final model
        if n < 0:
            m = net_func()
            m.load_state_dict(copy.deepcopy(model.state_dict()))
            m.to("cpu")
            list_of_sgd_models.append(m)
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val).item()
                main_losses.append(val_loss)
                test_pred = (model(x_val) > 0).float()
                test_acc = (test_pred == y_val).float().mean().item()
                test_accuracies.append(test_acc)
                logger.debug(
                    f"Final Validation Loss: {val_loss:.4f}, Final Test Accuracy: {test_acc:.4f}"
                )

        elif compute_counterfactual:
            m = net_func()
            m.load_state_dict(copy.deepcopy(model.state_dict()))
            m.to("cpu")
            list_of_counterfactual_models[n].models.append(m)

        # Clean up after each iteration
        del model
        torch.cuda.empty_cache()

    # Save more detailed information
    data_to_save = {
        "models": NetList(list_of_sgd_models),
        # models: NetList object containing (num_epoch * num_steps + 1) models
        # Each model's shape depends on the model_type (logreg, dnn, or cnn)
        "info": info,
        # info: List of dictionaries, length = (num_epoch * num_steps)
        # Each dict contains 'idx' (array of integers) and 'lr' (float)
        "counterfactual": list_of_counterfactual_models,
        # counterfactual: List of NetList objects if compute_counterfactual is True, else None
        # Length = n_tr if compute_counterfactual is True
        # Each NetList contains (num_epoch + 1) models
        "alpha": alpha,
        # alpha: float, regularization parameter
        "main_losses": main_losses,
        # main_losses: List of floats, length = (num_epoch + 1)
        # Contains validation losses at the end of each epoch
        "test_accuracies": test_accuracies,
        # test_accuracies: List of floats, length = (num_epoch + 1)
        # Contains test accuracies at the end of each epoch
        "train_losses": train_losses,
        # train_losses: numpy array of shape (num_epoch * num_steps + 1,)
        # Contains training losses for each batch
        "seed": seed,
        # seed: integer, random seed used
        "n_tr": data_sizes["n_tr"],
        # n_tr: integer, number of training samples
        "n_val": data_sizes["n_val"],
        # n_val: integer, number of validation samples
        "n_test": data_sizes["n_test"],
        # n_test: integer, number of test samples
        "num_epoch": training_params["num_epoch"],
        # num_epoch: integer, number of training epochs
        "batch_size": training_params["batch_size"],
        # batch_size: integer, size of each training batch
        "lr": training_params["lr"],
        # lr: float, initial learning rate
        "decay": training_params["decay"],
        # decay: boolean, whether learning rate decay is applied
    }

    # Save data
    torch.save(data_to_save, fn)

    # info.insert(0, {"idx": np.arange(data_sizes["n_tr"]), "lr": training_params["lr"]})

    # save step and info
    step_fn_csv = os.path.join(dn, f"step_{seed:03d}.csv")

    logger.debug(
        f"len('step') = {len(range(len(info)))}, len('lr') = {len([d['lr'] for d in info])}, len('idx') = {len([d['idx'] for d in info])}"
    )

    pd.DataFrame(
        {
            "step": range(len(info)),
            "lr": [d["lr"] for d in info],
            "idx": [d["idx"] for d in info],
        }
    ).to_csv(step_fn_csv, index=False)

    # Save main_losses and test_accuracies to CSV
    csv_fn = os.path.join(dn, f"metrics_{seed:03d}.csv")
    pd.DataFrame(
        {
            "epoch": range(len(main_losses)),
            "main_loss": main_losses,
            "test_accuracy": test_accuracies,
            "train_loss": train_losses,
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
        custom_n_tr=args.n_tr or default_config.get("n_tr"),
        custom_n_val=args.n_val or default_config.get("n_val"),
        custom_n_test=args.n_test or default_config.get("n_test"),
        custom_num_epoch=args.num_epoch or default_config.get("num_epoch"),
        custom_batch_size=args.batch_size or default_config.get("batch_size"),
        custom_lr=args.lr or default_config.get("lr"),
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
