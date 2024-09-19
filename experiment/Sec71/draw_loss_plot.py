###
# File: /draw_loss_plot.py
# Created Date: Thursday, September 12th 2024
# Author: Zihan
# -----
# Last Modified: Sunday, 15th September 2024 5:28:30 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import numpy as np
import torch
from matplotlib import pyplot as plt
import sys
import os
from multiprocessing import Pool
import argparse

# 导入自定义模块
import MyNet
from MyNet import NetList, LogReg, DNN, CifarCNN
import warnings

# no future warning
warnings.simplefilter(action="ignore", category=FutureWarning)
# 将自定义类添加到全局命名空间
sys.modules["MyNet"] = sys.modules[__name__]


def load_data(file_path: str, device: str = "cpu"):
    try:
        data = torch.load(file_path, map_location=device)

        # 处理模型
        if isinstance(data.get("models"), NetList):
            data["models"] = NetList(
                [model.to(device) for model in data["models"].models]
            )

        # 处理反事实模型
        if isinstance(data.get("counterfactual"), list) and all(
            isinstance(item, NetList) for item in data["counterfactual"]
        ):
            data["counterfactual"] = [
                NetList([model.to(device) for model in netlist.models])
                for netlist in data["counterfactual"]
            ]

        return data
    except Exception as e:
        print(f"无法加载数据: {file_path}")
        print(f"错误: {e}")
        return None


def get_model_type(model):
    if isinstance(model, LogReg):
        return "LogReg"
    elif isinstance(model, DNN):
        return "DNN"
    elif isinstance(model, CifarCNN):
        return "CifarCNN"
    else:
        return "Unknown"


def generate_loss_plot(args):
    file_path, device = args
    res = load_data(file_path, device)
    if res is None:
        print(f"无法加载数据: {file_path}")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate epochs and steps
    steps = len(res["info"])
    epochs = len(res["main_losses"]) - 1
    steps_per_epoch = steps / epochs

    # Training loss plot
    x_train = np.arange(len(res["train_losses"])) / steps_per_epoch
    ax.plot(x_train, res["train_losses"], label="Training Loss", alpha=0.7)

    # Main model loss plot
    x_main = np.arange(len(res["main_losses"]))
    ax.plot(x_main, res["main_losses"], label="Main Model Loss", alpha=0.7)

    # Get model type
    model_type = get_model_type(res["models"].models[0])

    # Set labels and title
    ax.set_title(
        f"Losses vs. Epochs - {os.path.basename(file_path)} ({model_type})", fontsize=16
    )
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=10)

    # Add text with additional information
    info_text = (
        f"Model Type: {model_type}\n"
        f"Total Epochs: {epochs:.1f}\n"
        f"Steps per Epoch: {steps_per_epoch:.0f}\n"
        f"Batch Size: {len(res['info'][0]['idx'])}\n"
        f"Alpha: {res['alpha']}"
    )
    ax.text(
        0.95,
        0.95,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    save_path = file_path.replace(".dat", f"_loss_{model_type}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss plot saved to: {save_path}")


def main(
    base_path: str, num_cpus: int, device: str, initial_seed: int, final_seed: int
):
    file_paths = [
        os.path.join(base_path, f"sgd{i:03d}.dat")
        for i in range(initial_seed, final_seed + 1)
    ]
    print(f"处理路径: {base_path}")
    print(f"使用 {num_cpus} 个 CPU 核心处理数据...")
    print(f"使用设备: {device}")
    print(f"处理种子范围: {initial_seed} 到 {final_seed}")

    # 使用指定数量的核心并行处理
    with Pool(num_cpus) as p:
        p.map(generate_loss_plot, [(fp, device) for fp in file_paths])

    print("所有损失图生成完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并行生成模型损失图")
    parser.add_argument(
        "--path", type=str, required=True, help="数据文件所在的基础路径"
    )
    parser.add_argument(
        "--cpus", type=int, default=16, help="使用的CPU核心数（默认：16）"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="使用的设备（默认：cpu）"
    )
    parser.add_argument(
        "--initial_seed", type=int, default=0, help="起始种子值（默认：0）"
    )
    parser.add_argument(
        "--final_seed", type=int, default=99, help="结束种子值（默认：99）"
    )
    args = parser.parse_args()

    main(args.path, args.cpus, args.device, args.initial_seed, args.final_seed)

# 运行示例命令：
# python draw_loss_plot.py --path /home/zihan/codes/sgd-influence/experiment/Sec71/cifar_cnn/ --cpus 8 --device cuda:0 --initial_seed 98 --final_seed 99
