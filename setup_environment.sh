#!/bin/bash

# 定义环境路径
ENV_PATH="$PWD/.venv"

# 创建一个新的 Conda 虚拟环境
conda create -p $ENV_PATH python=3.11 -y

# 激活虚拟环境
conda activate $ENV_PATH

# 安装基础数据科学包
conda install numpy pandas matplotlib -y

# 安装 PyTorch 及相关包 (假设你需要使用 CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 tensorflow -c pytorch -c nvidia -y

# 安装 scikit-learn
conda install scikit-learn -y

# 打印安装的 Python 和 pip 版本
python --version
pip --version

echo "环境设置完成，并已安装所需的包。"

