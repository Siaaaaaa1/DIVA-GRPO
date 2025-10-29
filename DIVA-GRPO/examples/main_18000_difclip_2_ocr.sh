#!/bin/bash

set -x

EXP_NAME="main_difclip_var2-$(date +%m%d-%H)"

MODEL_PATH="/models/Qwen2.5-VL-7B-Instruct" # replace it with your local file path
LOG_PATH="logs/"

if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
fi

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/datasets/train_dataset.parquet \
    data.val_files=/datasets/MMK12_test.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=8 \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.total_epochs=5 \
    trainer.val_freq=1 \
    trainer.val_generations_to_log=1 \
    trainer.save_limit=10 \
    trainer.save_freq=1 \
    trainer.Difficulty_Adaptation=true \
    trainer.Varient_Num=1 \
    data.max_pixels=1048576 \
    data.shuffle=true \
    trainer.Difficulty_Change=true \
    worker.rollout.disable_tqdm=false \
    trainer.val_before_train=true \
    algorithm.Adjust_Low_Reward_Local=true \
    algorithm.Adjust_Low_Reward_Global=true \
    algorithm.weight_mode="weightafter1-5_zscore_norm" \
    trainer.Dataset_Mode="only_text_thinking" \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false
    2>&1 | tee -a "${LOG_PATH}/${EXP_NAME}_training_log.log"
wait