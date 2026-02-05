#!/bin/bash
# PPO Training script for GSM8K on single GPU
# Usage: bash jl_patch/run_ppo_training.sh

export WANDB_API_KEY="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"

set -e

# Set paths
DATA_DIR="/mnt/hdfs/tiktok_aiic/user/junlongli/data/gsm8k"
MODEL_DIR="/mnt/hdfs/tiktok_aiic/user/junlongli/models/Qwen2.5-0.5B-Instruct"
CHECKPOINT_DIR="/mnt/hdfs/tiktok_aiic/user/junlongli/checkpoints/verl_quickstart"

# wandb settings (customize if needed)
PROJECT_NAME="verl-gsm8k"
EXPERIMENT_NAME="ppo-qwen2.5-0.5b-single-gpu"

echo "Starting PPO training..."
echo "Data directory: $DATA_DIR"
echo "Model directory: $MODEL_DIR"
echo "WandB project: $PROJECT_NAME"
echo "WandB experiment: $EXPERIMENT_NAME"
echo "----------------------------------------"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$MODEL_DIR \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=$MODEL_DIR \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee jl_patch/verl_training.log

echo "----------------------------------------"
echo "Training completed! Log saved to jl_patch/verl_training.log"
