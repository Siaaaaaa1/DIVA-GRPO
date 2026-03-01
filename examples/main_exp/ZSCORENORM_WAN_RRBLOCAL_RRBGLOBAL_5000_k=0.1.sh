#!/bin/bash

set -x

SCRIPT_NAME=$(basename "$0" .sh)
EXP_NAME="${SCRIPT_NAME}_$(date +%m%d-%H%M)"

# Environment Variables
export PYTHONUNBUFFERED=1
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=Easy-R1
export WANDB_RUN_NAME=${EXP_NAME}--$(date +%Y-%m-%d-%H-%M-%S)
# Ensure WANDB_API_KEY is set in your environment variables
# export WANDB_API_KEY="YOUR_API_KEY_HERE"
# wandb login $WANDB_API_KEY

# Define Relative Paths
# Please ensure these directories exist relative to where you run the script
MODEL_PATH="models/Qwen2.5-VL-7B-Instruct"
LOG_PATH="logs"
ALL_LOG_PATH="${LOG_PATH}/all_log_path"
DATA_TRAIN="datasets/train.parquet"
DATA_VAL="datasets/test.parquet"

# Create Log Directories
mkdir -p "$LOG_PATH"
mkdir -p "${ALL_LOG_PATH}/${SCRIPT_NAME}"

# Run Training
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${DATA_TRAIN}" \
    data.val_files="${DATA_VAL}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    trainer.n_gpus_per_node=8 \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.total_epochs=5 \
    trainer.val_freq=10 \
    trainer.val_generations_to_log=1 \
    trainer.save_limit=10 \
    trainer.save_freq=1 \
    trainer.val_before_train=false \
    trainer.find_last_checkpoint=false \
    data.rollout_batch_size=128 \
    worker.actor.global_batch_size=128 \
    trainer.All_Log_Path="${ALL_LOG_PATH}/${EXP_NAME}" \
    data.max_pixels=1048576 \
    worker.rollout.disable_tqdm=false \
    data.shuffle=false \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    trainer.val_before_train=true \
    trainer.DIVA_GRPO=true \
    trainer.Share_VL=false \
    trainer.Variant_Num=1 \
    trainer.Dataset_Mode=None \
    trainer.Save_Data=false \
    algorithm.weight_mode="${SCRIPT_NAME}$" \
    trainer.score_ranges='[[0.0,0.05],[0.05,0.3],[0.3,0.5],[0.5,0.7],[0.7,1.0]]' \
    trainer.difficulty_changes='[2,1,0,-1,-2]' \
    trainer.weighted_advantage_k=0.1 \
    worker.reward.reward_save_path="${ALL_LOG_PATH}/${EXP_NAME}" \
    2>&1 | tee -a "${LOG_PATH}/${EXP_NAME}_training_log.log"
wait