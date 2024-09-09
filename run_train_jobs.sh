#!/bin/bash
#SBATCH --job-name=train_dnn_multi_gpu
#SBATCH --output=train_output_%j.log
#SBATCH --error=train_error_%j.log
#SBATCH --nodelist=node9
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

# 加载需要的模块
module load cuda/11.3

# 设置环境变量
export CONDA_PREFIX="/home/zihanwu7/sgd-influence/.venv"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PATH="/usr/local/cuda-11.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"

# 切换到指定的工作目录
cd /home/zihanwu7/sgd-influence/experiment/Sec71

# 记录总开始时间
total_start_time=$(date +%s)

# 设置参数
n_gpus=4
m1=0
m2=99

# 计算每个GPU处理的seed数量
seeds_per_gpu=$(( (m2 - m1 + 1 + n_gpus - 1) / n_gpus ))

# 定义函数来运行实验
run_experiment() {
    local gpu=$1
    local start_seed=$2
    local end_seed=$3

    export MY_PYTHON="/home/zihanwu7/sgd-influence/.venv/bin/python"
    
    for ((seed=start_seed; seed<=end_seed && seed<=m2; seed++)); do
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        echo "[$timestamp][gpu_$gpu] Running with seed=$seed"
        start_time=$(date +%s)

        export CUDA_VISIBLE_DEVICES=$gpu
        
        # $MY_PYTHON train.py --target adult --model dnn --seed $seed --gpu 0 --trainer new
        # $MY_PYTHON train.py --target adult --model dnn --seed $seed --gpu 0
        # $MY_PYTHON changed_from_origin_train.py --target adult --model dnn --seed $seed --gpu 0
        # $MY_PYTHON infl.py --target adult --model dnn --type true --seed $seed --gpu 0
        $MY_PYTHON infl.py --target adult --model dnn --type sgd --seed $seed --gpu 0
        # $MY_PYTHON infl.py --target adult --model dnn --type icml --seed $seed --gpu 0
        # $MY_PYTHON infl.py --target adult --model dnn --type lie --gpu 0 --seed $seed
        
        if [ $? -ne 0 ]; then
            timestamp=$(date +"%Y-%m-%d %H:%M:%S")
            echo "[$timestamp][gpu_$gpu] Error occurred with seed=$seed. Exiting."
            exit 1
        fi
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        echo "[$timestamp][gpu_$gpu] Task for seed $seed completed in $duration seconds"
    done
}

# 并行运行n个GPU的实验
for ((i=0; i<n_gpus; i++)); do
    start_seed=$((m1 + i * seeds_per_gpu))
    end_seed=$((start_seed + seeds_per_gpu - 1))
    run_experiment $i $start_seed $end_seed &
done

# 等待所有后台任务完成
wait

# 计算并输出所有任务的总执行时间
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$timestamp][main] All iterations completed successfully."
echo "[$timestamp][main] Total execution time: $total_duration seconds"
