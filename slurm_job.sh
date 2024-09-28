#!/bin/zsh
#SBATCH --job-name=mnist_cleansing            # 作业名称
#SBATCH --output=%x_%j.log                   # 输出文件
#SBATCH --ntasks=1                           # 总任务数
#SBATCH --cpus-per-task=6                    # 每个任务所需的 CPU 核数
#SBATCH --gres=gpu:3                         # 需要的 GPU 数量
#SBATCH --mem=64G                            # 内存
#SBATCH --time=48:00:00                      # 作业的最大运行时间
#SBATCH --partition=debug                    # 使用的分区

# 设置初始seed和终了seed
INITIAL_SEED=0
FINAL_SEED=15

# 默认设置
HOME_DIR="/home/zihan/codes"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --runpod)
      HOME_DIR="/workspace"
      shift # 移动到下一个参数
      ;;
    --local_linux)
      HOME_DIR="/home/zihan/codes"
      shift # 移动到下一个参数
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 设置其他路径
PYTHON_ENV="$HOME_DIR/sgd-influence/.venv/bin/python"
WORK_DIR="$HOME_DIR/sgd-influence/experiment/Sec71"
TRAIN_SCRIPT="$WORK_DIR/train.py"
CLEANSING_SCRIPT="$WORK_DIR/data_cleansing.py"
INFL_SCRIPT="$WORK_DIR/infl.py"

PYTHON_COMMAND='
    for model in logreg dnn cnn; do
        for relabel in 20 30 10; do
            for check in 5 10 15 20 25 30 35 40 45 50; do
                # 训练模型
                $PYTHON_ENV "$TRAIN_SCRIPT" --target mnist --model "$model" --seed "$seed" --gpu 0 --save_dir cleansing/mnist_"$model"_relabel_"$relabel" --no-loo --relabel "$relabel";
                
                # 运行影响计算
                for type in true sgd icml segment_true lie; do
                    $PYTHON_ENV "$INFL_SCRIPT" --target mnist --model "$model" --seed "$seed" --gpu 0 --save_dir cleansing/mnist_"$model"_relabel_"$relabel" --relabel "$relabel" --type "$type";
                    $PYTHON_ENV "$CLEANSING_SCRIPT" --target mnist --model "$model" --seed "$seed" --gpu 0 --save_dir cleansing/mnist_"$model"_relabel_"$relabel" --relabel "$relabel" --type "$type" --check "$check";
                done

                for type in dit_first dit_middle dit_last true_first true_middle true_last; do
                    $PYTHON_ENV "$CLEANSING_SCRIPT" --target mnist --model "$model" --seed "$seed" --gpu 0 --save_dir cleansing/mnist_"$model"_relabel_"$relabel" --relabel "$relabel" --type "$type" --check "$check";
                done

                # 数据清理
            done
        done
    done
'

# 打印当前设置（用于调试）
echo "Current settings:"
echo "HOME_DIR: $HOME_DIR"
echo "PYTHON_ENV: $PYTHON_ENV"
echo "WORK_DIR: $WORK_DIR"
echo "TRAIN_SCRIPT: $TRAIN_SCRIPT"
echo "INFL_SCRIPT: $INFL_SCRIPT"
echo "CLEANSING_SCRIPT: $CLEANSING_SCRIPT"

# 显示调试信息
echo "========== Debug Info =========="
echo "Current working directory: $(pwd)"
echo "Python interpreter: $PYTHON_ENV"
echo "Initial seed: $INITIAL_SEED"
echo "Final seed: $FINAL_SEED"
echo "==============================="

# 切换到工作目录
cd "$WORK_DIR" || exit 1

echo "========== Job Info =========="
echo "Job started at: $(date)"
echo "Job ID: $$"
echo "Node list: $SLURM_JOB_NODELIST"
echo "GPUs: All available"
echo "==============================="

# 获取可用的GPU数量
n_gpus=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $n_gpus"

# 创建一个临时文件来存储下一个种子值
SEED_FILE="/tmp/next_seed_$$"
echo $INITIAL_SEED > $SEED_FILE

# 定义一个函数来安全地获取下一个种子值
get_next_seed() {
    local next_seed
    local lock_file="/tmp/seed_lock_$$"

    # 尝试获取锁
    while ! mkdir "$lock_file" 2>/dev/null; do
        sleep 0.1
    done

    # 读取和更新种子值
    next_seed=$(<$SEED_FILE)
    echo $((next_seed + 1)) > $SEED_FILE

    # 释放锁
    rmdir "$lock_file"

    echo $next_seed
}

run_experiment() {
    local seed="$1"
    local gpu="$2"

    if [[ -z "$seed" || -z "$gpu" ]]; then
        echo "Error: seed or gpu parameter not set"
        return 1
    fi

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp][gpu_$gpu] Running with seed=$seed"

    if [ ! -f "$PYTHON_ENV" ]; then
        echo "Python interpreter not found at $PYTHON_ENV"
        return 1
    fi

    if [ ! -f "$TRAIN_SCRIPT" ]; then
        echo "Train script not found at $TRAIN_SCRIPT"
        return 1
    fi

    # Set environment variables to limit GPU memory usage
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

    # 使用变量执行 Python 命令
    eval $PYTHON_COMMAND

    local exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "[$timestamp][gpu_$gpu] Error occurred with seed=$seed. Exit status: $exit_status"
        return 1
    fi

    echo "[$timestamp][gpu_$gpu] Task for seed $seed completed"
}

# 每个GPU上运行的最大并发进程数
max_processes_per_gpu=2

# 创建一个函数来处理每个GPU的任务
process_gpu_tasks() {
    local gpu=$1
    local pids=()

    while true; do
        # 检查当前运行的进程数
        while (( ${#pids} >= max_processes_per_gpu )); do
            for pid in $pids; do
                if ! kill -0 $pid 2>/dev/null; then
                    pids=("${(@)pids:#$pid}")
                fi
            done
            sleep 1
        done

        # 获取下一个种子
        local seed=$(get_next_seed)

        # 检查是否还有未处理的种子
        if (( seed > FINAL_SEED )); then
            break
        fi

        run_experiment "$seed" "$gpu" &
        pids+=($!)
    done

    # 等待该GPU上的所有进程完成
    for pid in $pids; do
        wait $pid
    done
}

total_start_time=$(date +%s)

# 为每个GPU启动一个后台进程来处理任务
for ((gpu=0; gpu<n_gpus; gpu++)); do
    process_gpu_tasks $gpu &
done

# 等待所有GPU任务处理完毕
wait

# 清理临时文件
rm -f $SEED_FILE /tmp/seed_lock_$$

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$timestamp][main] All iterations completed."
echo "[$timestamp][main] Total execution time: $total_duration seconds"
