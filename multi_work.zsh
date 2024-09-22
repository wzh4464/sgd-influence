#!/bin/zsh
# Set zsh options for better script behavior
setopt ERR_EXIT
setopt NO_UNSET
setopt PIPE_FAIL

# 设置初始seed和终了seed
INITIAL_SEED=0
FINAL_SEED=15

# 设置绝对路径
# HOME_DIR="/workspace"
HOME_DIR="/home/zihan/codes"
PYTHON_ENV="$HOME_DIR/sgd-influence/.venv/bin/python"
WORK_DIR="$HOME_DIR/sgd-influence/experiment/Sec71"
TRAIN_SCRIPT="$WORK_DIR/train.py"

# PYTHON_COMMAND='
# #     $PYTHON_ENV "$WORK_DIR/train.py" --target mnist --model logreg --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model logreg --type true --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model logreg --type sgd --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model logreg --type icml --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model logreg --type lie --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model logreg --type segment_true --seed "$seed" --gpu 0;
#     $PYTHON_ENV "$WORK_DIR/train.py" --target mnist --model dnn --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model dnn --type true --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model dnn --type sgd --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model dnn --type icml --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model dnn --type lie --seed "$seed" --gpu 0;
#     $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model dnn --type segment_true --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/train.py" --target 20news --model logreg --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target 20news --model logreg --type true --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target 20news --model logreg --type sgd --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target 20news --model logreg --type icml --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target 20news --model logreg --type lie --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target 20news --model logreg --type segment_true --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/train.py" --target 20news --model dnn --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target 20news --model dnn --type true --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target 20news --model dnn --type sgd --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target 20news --model dnn --type icml --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target 20news --model dnn --type lie --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target 20news --model dnn --type segment_true --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/train.py" --target adult --model logreg --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target adult --model logreg --type true --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target adult --model logreg --type sgd --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target adult --model logreg --type icml --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target adult --model logreg --type lie --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target adult --model logreg --type segment_true --seed "$seed" --gpu 0;
#     $PYTHON_ENV "$WORK_DIR/train.py" --target adult --model dnn --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target adult --model dnn --type true --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target adult --model dnn --type sgd --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target adult --model dnn --type icml --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target adult --model dnn --type lie --seed "$seed" --gpu 0;
#     $PYTHON_ENV "$WORK_DIR/infl.py" --target adult --model dnn --type segment_true --seed "$seed" --gpu 0;
#     $PYTHON_ENV "$WORK_DIR/train.py" --target emnist --model dnn --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target emnist --model dnn --type true --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target emnist --model dnn --type sgd --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target emnist --model dnn --type icml --seed "$seed" --gpu 0;
# #     $PYTHON_ENV "$WORK_DIR/infl.py" --target emnist --model dnn --type lie --seed "$seed" --gpu 0;
#     $PYTHON_ENV "$WORK_DIR/infl.py" --target emnist --model dnn --type segment_true --seed "$seed" --gpu 0;
# '

PYTHON_COMMAND='
    $PYTHON_ENV "$WORK_DIR/train.py" --target mnist --model cnn --seed "$seed" --gpu 0;
    $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model cnn --type true --seed "$seed" --gpu 0;
    $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model cnn --type sgd --seed "$seed" --gpu 0;
    $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model cnn --type icml --seed "$seed" --gpu 0;
    $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model cnn --type lie --seed "$seed" --gpu 0;
    $PYTHON_ENV "$WORK_DIR/infl.py" --target mnist --model cnn --type segment_true --gpu 0;
'

# 显示调试信息
echo "========== Debug Info =========="
echo "Current working directory: $(pwd)"
echo "Python interpreter: $PYTHON_ENV"
echo "Python command: $PYTHON_COMMAND"
echo "Initial seed: $INITIAL_SEED"
echo "Final seed: $FINAL_SEED"
echo "==============================="

# 切换到工作目录
cd "$WORK_DIR" || exit 1

echo "========== Job Info =========="
echo "Job started at: $(date)"
echo "Job ID: $$"
echo "Node list: localhost"
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
