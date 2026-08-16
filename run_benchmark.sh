#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --array=0-11%4
#SBATCH --array=0
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH --output=%x_%j_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

echo "## Available CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

datasets=(
    "dcase"
    "watkins"
    "bats"
    "dogs"
    "cbi"
    "humbugdb"
    "rfcx"
    "enabirds"
    "hiceas"
    "hainan-gibbons"
    "speech-commands"
    "esc50"
    )

tasks=(
    "detection"
    "classification"
    "classification"
    "classification"
    "classification"
    "classification"
    "detection"
    "detection"
    "detection"
    "detection"
    "classification"
    "classification"
    )
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
task=${tasks[$SLURM_ARRAY_TASK_ID]}
log_path="logs/${dataset}-${MODEL_TYPE}"
echo "## Evaluating on dataset: $dataset and model type: $MODEL_TYPE with the following parameters: batch size: $BATCH_SIZE, learning rates: $LRS, task: $task, log path: $log_path, number of workers: $NUM_WORKERS"
cd $HOME/github/beans
source .venv/bin/activate
python -m scripts.evaluate --model-type $MODEL_TYPE --lrs $LRS --task $task --dataset $dataset --batch-size $BATCH_SIZE --log-path $log_path --num-workers $NUM_WORKERS
