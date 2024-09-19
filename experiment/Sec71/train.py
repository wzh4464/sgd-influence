###
# File: experiment/Sec71/train.py
# Created Date: September 9th 2024
# Author: Zihan
# -----
# Last Modified: Thursday, 19th September 2024 9:39:09 pm
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
import random

# Assuming these imports are from local files
from DataModule import MnistModule, NewsModule, AdultModule, CifarModule, EMNISTModule
from MyNet import LogReg, DNN, NetList, CifarCNN

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

file_abspath = os.path.abspath(__file__)
current_dir = os.path.dirname(file_abspath)  # 获取当前脚本所在的目录路径


def get_data_module(
    key: str, csv_path: str
) -> Tuple[Any, Dict[str, int], Dict[str, Any]]:
    if key == "mnist":
        module = MnistModule(data_dir=csv_path)
        data_sizes = {"n_tr": 200, "n_val": 200, "n_test": 200}
        training_params = {
            "num_epoch": 21,
            "batch_size": 60,
            "lr": 0.003,
            "decay": True,
        }
    elif key == "20news":
        module = NewsModule(data_dir=csv_path)
        data_sizes = {"n_tr": 200, "n_val": 200, "n_test": 200}
        training_params = {"num_epoch": 21, "batch_size": 60, "lr": 0.01, "decay": True}
    elif key == "adult":
        module = AdultModule(csv_path=csv_path)
        data_sizes = {"n_tr": 200, "n_val": 1000, "200": 200}
        training_params = {"num_epoch": 21, "batch_size": 60, "lr": 0.01, "decay": True}
    elif key == "cifar":
        module = CifarModule(data_dir=csv_path)
        data_sizes = {"n_tr": 200, "n_val": 200, "n_test": 200}
        training_params = {"num_epoch": 21, "batch_size": 60, "lr": 0.01, "decay": True}
    elif key == "emnist":  # Add this new condition
        module = EMNISTModule(data_dir=csv_path)
        data_sizes = {"n_tr": 200, "n_val": 200, "n_test": 200}
        training_params = {"num_epoch": 21, "batch_size": 60, "lr": 0.2, "decay": True}
    else:
        raise ValueError(f"Unsupported dataset: {key}")

    module.append_one = False
    return module, data_sizes, training_params


def get_model(model_type: str, input_dim: int, device: str) -> nn.Module:
    if model_type == "logreg":
        return LogReg(input_dim).to(device)
    elif model_type == "dnn":
        return DNN(input_dim).to(device)
    elif model_type == "cnn":
        return CifarCNN().to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


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
    compute_counterfactual: bool = True,
    logger=None,
) -> Dict[str, Any]:
    if csv_path is None:
        csv_path = os.path.join(current_dir, "data")

    # 创建存储模型的目录，基于当前脚本路径
    dn = os.path.join(current_dir, f"{key}_{model_type}")
    fn = os.path.join(dn, f"sgd{seed:03d}.dat")
    os.makedirs(dn, exist_ok=True)

    device = f"cuda:{gpu}"

    # Fetch data and settings
    module, data_sizes, training_params = get_data_module(key, csv_path)

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

    z_tr, z_val, _ = module.fetch(
        data_sizes["n_tr"], data_sizes["n_val"], data_sizes["n_test"], seed
    )
    (x_tr, y_tr), (x_val, y_val) = z_tr, z_val

    logger.info(
        f"Dataset {key} loaded with {data_sizes['n_tr']} training samples, {data_sizes['n_val']} validation samples"
    )

    # Model selection and hyperparameter tuning
    if model_type == "logreg":
        model = LogisticRegressionCV(random_state=seed, fit_intercept=False, cv=5)
        model.fit(x_tr, y_tr)
        alpha = 1 / (model.C_[0] * data_sizes["n_tr"])
    elif model_type in {"dnn", "cnn"}:
        alpha = 0.001  # You might want to tune this for DNN/CNN
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Model {model_type} initialized with alpha={alpha}")

    # Convert to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)

    # Reshape for CNN if necessary
    if model_type == "cnn":
        x_tr = x_tr.view(-1, 3, 32, 32)
        x_val = x_val.view(-1, 3, 32, 32)

    # Training setup
    net_func = lambda: get_model(model_type, x_tr.shape[1], device)
    num_steps = int(np.ceil(data_sizes["n_tr"] / training_params["batch_size"]))
    list_of_sgd_models = []
    list_of_counterfactual_models = (
        [NetList([]) for _ in range(data_sizes["n_tr"])]
        if compute_counterfactual
        else None
    )
    main_losses = []
    test_accuracies = []
    train_losses = np.zeros(training_params["num_epoch"] * num_steps + 1)

    logger.info(f"Starting training for {training_params['num_epoch']} epochs")

    # Training loop
    for n in range(-1, data_sizes["n_tr"] if compute_counterfactual else 0):
        torch.manual_seed(seed)
        model = net_func()
        loss_fn = nn.BCEWithLogitsLoss()
        # Training setup
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.0,
            weight_decay=1e-4,  # 学习率变小，加入L2正则化
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
                    m.to("cpu")  # Move model to CPU memory
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
                            logger.info(
                                f"Epoch {epoch+1}/{training_params['num_epoch']}, Validation Loss: {val_loss:.4f}, Test Accuracy: {test_acc:.4f}"
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
                z = model(x_tr[idx])
                loss = loss_fn(z, y_tr[idx])

                train_losses[c] = loss.item()
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
            logger.info(
                f"Epoch {epoch+1}/{training_params['num_epoch']}, Average Training Loss: {epoch_loss/num_steps:.4f}"
            )
            torch.cuda.empty_cache()

        # Save final model
        if n < 0:
            m = net_func()
            m.load_state_dict(copy.deepcopy(model.state_dict()))
            m.to("cpu")  # Move model to CPU memory
            list_of_sgd_models.append(m)
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val).item()
                main_losses.append(val_loss)
                test_pred = (model(x_val) > 0).float()
                test_acc = (test_pred == y_val).float().mean().item()
                test_accuracies.append(test_acc)
                logger.info(
                    f"Final Validation Loss: {val_loss:.4f}, Final Test Accuracy: {test_acc:.4f}"
                )

        elif compute_counterfactual:
            m = net_func()
            m.load_state_dict(copy.deepcopy(model.state_dict()))
            m.to("cpu")  # Move model to CPU memory
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

    # Save main_losses and test_accuracies to CSV
    csv_fn = os.path.join(dn, f"metrics_{seed:03d}.csv")
    pd.DataFrame(
        {
            "epoch": range(len(main_losses)),
            "main_loss": main_losses,
            "test_accuracy": test_accuracies,
        }
    ).to_csv(csv_fn, index=False)

    logger.info(f"Training completed. Results saved to {fn} and {csv_fn}")

    return data_to_save


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
    parser.add_argument(
        "--no-loo",
        action="store_false",
        dest="compute_counterfactual",  # 设置为 False
        help="Disable the computation of counterfactual models (leave-one-out).",
    )

    # 默认 compute_counterfactual 为 True
    parser.set_defaults(compute_counterfactual=True)

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(f"{args.target}_{args.model}", args.seed)

    try:
        _validate_arguments(logger, args)
    except AssertionError as e:
        logger.error(f"Invalid argument: {str(e)}")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"An error occurred during the training process: {str(e)}")
        logger.error(traceback.format_exc())


def _validate_arguments(logger, args):
    logger.info("Starting the training process")
    logger.info(f"Arguments: {args}")

    assert args.target in [
        "mnist",
        "20news",
        "adult",
        "cifar",
        "emnist",
    ], "Invalid target data"
    assert args.model in ["logreg", "dnn", "cnn"], "Invalid model type"

    if args.seed >= 0:
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
            compute_counterfactual=args.compute_counterfactual,
            logger=logger,
        )
    else:
        for seed in range(100):
            train_and_save(
                args.target,
                args.model,
                seed,
                args.gpu,
                custom_n_tr=args.n_tr,
                custom_n_val=args.n_val,
                custom_n_test=args.n_test,
                custom_num_epoch=args.num_epoch,
                custom_batch_size=args.batch_size,
                compute_counterfactual=args.compute_counterfactual,
                logger=logger,
            )

    logger.info("Training process completed successfully")


if __name__ == "__main__":
    main()
