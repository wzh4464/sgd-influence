###
# File: /draw_loss_plot.py
# Created Date: Thursday, September 12th 2024
# Author: Zihan
# -----
# Last Modified: Thursday, 12th September 2024 3:51:38 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import numpy as np
import pandas as pd
import scipy.stats as stats
import joblib
import pickle
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import sys
from typing import Dict, Any
import os
from multiprocessing import Pool
import argparse

# 导入自定义模块
import MyNet
from MyNet import NetList, DNN

# 将自定义类添加到全局命名空间
sys.modules["MyNet"] = sys.modules[__name__]


def load_data(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"Joblib 加载失败: {e}")
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Pickle 加载失败: {e}")
            return None


def generate_loss_plot(file_path):
    res = load_data(file_path)
    if res is None:
        print(f"无法加载数据: {file_path}")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate epochs and steps
    epochs = len(res["info"]) / (200 / len(res["info"][0]["idx"]))
    steps_per_epoch = 200 / len(res["info"][0]["idx"])

    # Training loss plot
    x_train = np.arange(len(res["train_losses"])) / steps_per_epoch
    ax.plot(x_train, res["train_losses"], label="Training Loss", alpha=0.7)

    # Main model loss plot
    x_main = np.arange(len(res["main_losses"]))
    ax.plot(x_main, res["main_losses"], label="Main Model Loss", alpha=0.7)

    # Set labels and title
    ax.set_title(f"Losses vs. Epochs - {os.path.basename(file_path)}", fontsize=16)
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=10)

    # Add text with additional information
    info_text = (
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
    save_path = file_path.replace(".dat", "_loss.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss plot saved to: {save_path}")


def main(base_path, num_cpus):
    file_paths = [os.path.join(base_path, f"sgd{i:03d}.dat") for i in range(100)]

    print(f"处理路径: {base_path}")
    print(f"使用 {num_cpus} 个 CPU 核心处理数据...")

    # 使用指定数量的核心并行处理
    with Pool(num_cpus) as p:
        p.map(generate_loss_plot, file_paths)

    print("所有损失图生成完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并行生成模型损失图")
    parser.add_argument(
        "--path", type=str, required=True, help="数据文件所在的基础路径"
    )
    parser.add_argument(
        "--cpus", type=int, default=16, help="使用的CPU核心数（默认：16）"
    )
    args = parser.parse_args()

    main(args.path, args.cpus)

# 运行示例命令：
# python draw_loss_plot.py --path /home/zihan/codes/sgd-influence/experiment/Sec71/mnist_dnn/ --cpus 8
