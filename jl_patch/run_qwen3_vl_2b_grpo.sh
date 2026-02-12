#!/bin/bash
# Qwen3-VL-2B GRPO training script (Geo3K)
# Style aligned with jl_patch/run_ppo_training.sh
# Usage:
#   bash jl_patch/run_qwen3_vl_2b_grpo.sh
# Optional:
#   ENGINE=sglang bash jl_patch/run_qwen3_vl_2b_grpo.sh

export WANDB_API_KEY="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"


set -euo pipefail

# wandb settings
# If already exported in your shell, this line won't overwrite it.
PROJECT_NAME=${PROJECT_NAME:-"verl-geo3k"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo-qwen3-vl-2b"}

# Runtime settings
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

ENGINE=${ENGINE:-vllm}

# Paths (customized to your environment)
DATA_DIR=${DATA_DIR:-"/mnt/hdfs/tiktok_aiic/user/junlongli/data/geo3k"}
MODEL_DIR=${MODEL_DIR:-"/mnt/hdfs/tiktok_aiic/user/junlongli/models/Qwen/Qwen3-VL-2B-Instruct/"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/mnt/hdfs/tiktok_aiic/user/junlongli/checkpoints/verl_qwen3_vl_2b"}

# Parallelism (downsized from the 8B megatron example)
GEN_TP=${GEN_TP:-1}
CP=${CP:-1}
TP=${TP:-1}
PP=${PP:-1}

mkdir -p "$CHECKPOINT_DIR"

echo "Starting GRPO training (Qwen3-VL-2B)..."
echo "Engine: $ENGINE"
echo "Data directory: $DATA_DIR"
echo "Model directory: $MODEL_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "WandB project: $PROJECT_NAME"
echo "WandB experiment: $EXPERIMENT_NAME"
echo "----------------------------------------"

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path="$MODEL_DIR" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 "$@" 2>&1 | tee jl_patch/qwen3_vl_2b_grpo.log

echo "----------------------------------------"
echo "Training completed! Log saved to jl_patch/qwen3_vl_2b_grpo.log"
