#!/bin/zsh

# 设置 Python 环境和工作目录
PYTHON_ENV="/home/zihan/codes/sgd-influence/.venv/bin/python"
WORK_DIR="$HOME/codes/sgd-influence/experiment/Sec71"

# 设置默认值
TARGET="adult"
MODEL="logreg"
NUM_GPUS=3

# 解析命令行参数
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --target)
    TARGET="$2"
    shift # past argument
    shift # past value
    ;;
    --model)
    MODEL="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    echo "Unknown option: $1"
    exit 1
    ;;
esac
done

echo "Target: $TARGET"
echo "Model: $MODEL"
echo "Number of GPUs: $NUM_GPUS"
echo "Working Directory: $WORK_DIR"

# 切换到工作目录
cd $WORK_DIR || exit 1

# 定义一个函数来运行单个实验
run_experiment() {
    local seed=$1
    local gpu=$2
    echo "Running train.py with seed $seed on GPU $gpu"
    $PYTHON_ENV train.py --target $TARGET --model $MODEL --seed $seed --gpu $gpu
    
    echo "Running influence calculations with seed $seed on GPU $gpu"
    $PYTHON_ENV infl.py --target $TARGET --model $MODEL --type true --seed $seed --gpu $gpu
    $PYTHON_ENV infl.py --target $TARGET --model $MODEL --type sgd --seed $seed --gpu $gpu
    $PYTHON_ENV infl.py --target $TARGET --model $MODEL --type icml --seed $seed --gpu $gpu
}

# 并行运行实验
for seed in {0..99}
do
    gpu=$((seed % NUM_GPUS))
    run_experiment $seed $gpu &
    
    # 每启动 NUM_GPUS 个任务后等待它们完成
    if (( (seed + 1) % NUM_GPUS == 0 )); then
        wait
    fi
done

# 等待所有后台任务完成
wait

echo "All tasks completed."