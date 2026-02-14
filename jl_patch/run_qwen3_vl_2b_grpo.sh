#!/bin/bash
# Qwen3-VL-2B GRPO training script (Geo3K)
# Style aligned with jl_patch/run_ppo_training.sh
# Usage:
#   bash jl_patch/run_qwen3_vl_2b_grpo.sh
# Optional:
#   ENGINE=sglang bash jl_patch/run_qwen3_vl_2b_grpo.sh

set -euo pipefail

# ------------------------------
# Experiment identity / tracking
# ------------------------------
# If already exported in your shell, this line won't overwrite it.
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key_here}"
# W&B project name
PROJECT_NAME=${PROJECT_NAME:-"verl-geo3k"}
# W&B run name
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo-qwen3-vl-2b"}

# ------------------------------
# Runtime env
# ------------------------------
# Flush logs immediately
export PYTHONUNBUFFERED=1
# Better overlap for Megatron communication/computation
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Workaround for specific vLLM TP allreduce behavior
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# Verbose step-stage progress logs from trainer/worker (set 0 to disable)
export VERL_PROGRESS_LOG="${VERL_PROGRESS_LOG:-1}"
# Print every N global steps when verbose logging is enabled
export VERL_PROGRESS_LOG_INTERVAL="${VERL_PROGRESS_LOG_INTERVAL:-1}"
# Print actor update progress-bar every N mini-steps
export VERL_PROGRESS_BAR_INTERVAL="${VERL_PROGRESS_BAR_INTERVAL:-1}"
# Print rollout progress-bar every N finished samples (manager + selected worker)
export VERL_ROLLOUT_PROGRESS_INTERVAL="${VERL_ROLLOUT_PROGRESS_INTERVAL:-1}"
# Which agent-loop worker prints fine-grained sample progress (-1 means all workers)
export VERL_ROLLOUT_PROGRESS_WORKER_INDEX="${VERL_ROLLOUT_PROGRESS_WORKER_INDEX:-0}"
# Heartbeat period for long blocking phases like rollout generation
export VERL_PROGRESS_HEARTBEAT_SEC="${VERL_PROGRESS_HEARTBEAT_SEC:-5}"

# Rollout backend: vllm or sglang
ENGINE=${ENGINE:-vllm}

# ------------------------------
# Paths
# ------------------------------
# Geo3K parquet directory; expects train.parquet/test.parquet
DATA_DIR=${DATA_DIR:-"/mnt/hdfs/tiktok_aiic/user/junlongli/data/geo3k"}
# HF model path (local)
MODEL_DIR=${MODEL_DIR:-"/mnt/hdfs/tiktok_aiic/user/junlongli/models/Qwen/Qwen3-VL-2B-Instruct/"}
# Root directory for checkpoints and extra logs
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/mnt/hdfs/tiktok_aiic/user/junlongli/checkpoints/verl_qwen3_vl_2b"}
# Validation generations dump directory (jsonl)
VAL_GEN_DIR=${VAL_GEN_DIR:-"$CHECKPOINT_DIR/val_generations"}
# Train rollout generations dump directory (jsonl)
ROLLOUT_GEN_DIR=${ROLLOUT_GEN_DIR:-"$CHECKPOINT_DIR/rollout_generations"}

# ------------------------------
# Parallelism (downsized from 8B example)
# ------------------------------
# Rollout tensor parallel size (inference side)
GEN_TP=${GEN_TP:-1}
# Megatron context parallel size (training side)
CP=${CP:-1}
# Megatron tensor model parallel size (training side)
TP=${TP:-1}
# Megatron pipeline parallel size (training side)
PP=${PP:-1}

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$VAL_GEN_DIR"
mkdir -p "$ROLLOUT_GEN_DIR"

echo "Starting GRPO training (Qwen3-VL-2B)..."
echo "Engine: $ENGINE"
echo "Data directory: $DATA_DIR"
echo "Model directory: $MODEL_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Validation generations directory: $VAL_GEN_DIR"
echo "Rollout generations directory: $ROLLOUT_GEN_DIR"
echo "WandB project: $PROJECT_NAME"
echo "WandB experiment: $EXPERIMENT_NAME"
echo "----------------------------------------"

ARGS=(
    # -------- Base config --------
    --config-path=config
    --config-name=ppo_megatron_trainer.yaml

    # -------- Algorithm --------
    # Use GRPO advantage estimator
    algorithm.adv_estimator=grpo
    # Do not add KL in reward since KL loss is handled explicitly in actor loss
    algorithm.use_kl_in_reward=False

    # -------- Data --------
    # Train parquet
    data.train_files="$DATA_DIR/train.parquet"
    # Validation parquet
    data.val_files="$DATA_DIR/test.parquet"
    # Global prompt batch size per training step
    data.train_batch_size=128
    # Max prompt tokens
    data.max_prompt_length=1024
    # Max generated tokens
    data.max_response_length=1024
    # Drop overlong prompts
    data.filter_overlong_prompts=True
    # Fail when truncation would happen
    data.truncation=error
    # VLM image column name
    data.image_key=images

    # -------- Model --------
    # Actor/ref model path
    actor_rollout_ref.model.path="$MODEL_DIR"

    # -------- Actor optimization --------
    # Actor learning rate
    actor_rollout_ref.actor.optim.lr=1e-6
    # Global mini-batch size for actor update
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    # Per-GPU micro-batch size for actor update
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    # Enable KL loss in actor objective
    actor_rollout_ref.actor.use_kl_loss=True
    # KL loss coefficient
    actor_rollout_ref.actor.kl_loss_coef=0.01
    # KL estimator type
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    # Entropy bonus coefficient
    actor_rollout_ref.actor.entropy_coeff=0

    # -------- Megatron parallelism --------
    # Pipeline parallel size
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP
    # Tensor parallel size
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP
    # Context parallel size
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP
    # Enable mbridge for Megatron+VLM integration
    actor_rollout_ref.actor.megatron.use_mbridge=True

    # -------- Dynamic batching / token caps --------
    # Enable dynamic batching for actor update
    actor_rollout_ref.actor.use_dynamic_bsz=True
    # Max actor tokens per GPU under dynamic batching
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072
    # Enable dynamic batching for ref logprob
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    # Max ref logprob tokens per GPU
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072
    # Enable dynamic batching for rollout logprob
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    # Max rollout logprob tokens per GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072

    # -------- Rollout / inference --------
    # Rollout backend
    actor_rollout_ref.rollout.name=$ENGINE
    # Rollout TP size
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP
    # Per-GPU micro-batch for rollout logprob
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    # Per-GPU micro-batch for ref logprob
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    # vLLM target memory utilization
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    # GRPO group size: rollouts per prompt
    actor_rollout_ref.rollout.n=4
    # Disable vLLM multimodal preprocessor cache
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True

    # -------- Offload options (currently disabled) --------
    # Keep actor params on GPU
    actor_rollout_ref.actor.megatron.param_offload=False
    # Keep actor optimizer states on GPU
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    # Keep actor grads on GPU
    actor_rollout_ref.actor.megatron.grad_offload=False
    # Keep ref params on GPU
    actor_rollout_ref.ref.megatron.param_offload=False

    # -------- Activation recompute --------
    # Recompute strategy
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    # Recompute granularity
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    # Recompute layer count
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1

    # -------- Trainer / logging --------
    # Start actor update immediately (no critic warmup)
    trainer.critic_warmup=0
    # Logging backends
    trainer.logger=["console","wandb"]
    # W&B project name
    trainer.project_name="$PROJECT_NAME"
    # W&B run name
    trainer.experiment_name="$EXPERIMENT_NAME"
    # Checkpoint root directory
    trainer.default_local_dir="$CHECKPOINT_DIR"
    # Number of validation samples to log each validation step
    trainer.log_val_generations=20
    # Validation generations dump directory
    trainer.validation_data_dir="$VAL_GEN_DIR"
    # Training rollout generations dump directory
    trainer.rollout_data_dir="$ROLLOUT_GEN_DIR"

    # -------- Resources --------
    # GPUs per node
    trainer.n_gpus_per_node=1
    # Number of nodes
    trainer.nnodes=1

    # -------- Scheduling --------
    # Save checkpoint every N global steps
    trainer.save_freq=20
    # Run validation every N global steps
    trainer.test_freq=5
    # Total epochs
    trainer.total_epochs=10
)

python3 -m verl.trainer.main_ppo "${ARGS[@]}" "$@" 2>&1 | tee jl_patch/qwen3_vl_2b_grpo.log

echo "----------------------------------------"
echo "Training completed! Log saved to jl_patch/qwen3_vl_2b_grpo.log"
