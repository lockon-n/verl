# VERL 训练进度与 Rollout 进度条功能记录

更新时间：2026-02-12
范围：本次针对训练中间可观测性（progress logs / progress bars / rollout in-flight progress）所做的全部改动。

## 1. 目标

在不改训练算法行为（loss、梯度、权重更新逻辑）的前提下，提升训练过程可观测性：

- 在一个 global step 内看到分阶段进度（rollout/reward/old_log_prob/ref/adv/update_actor/update_weights/validate）。
- 在 actor 更新阶段看到 mini-step 级进度条。
- 在 async agent-loop rollout 阶段看到“已完成/总数”的实时进度条。
- 支持多机多卡下可控日志频率，避免日志洪泛。

## 2. 用法

最简用法：

```bash
bash jl_patch/run_qwen3_vl_2b_grpo.sh
```

按步降频 + rollout 降频（多机推荐）：

```bash
VERL_PROGRESS_LOG=1 \
VERL_PROGRESS_LOG_INTERVAL=2 \
VERL_PROGRESS_BAR_INTERVAL=2 \
VERL_ROLLOUT_PROGRESS_INTERVAL=8 \
VERL_ROLLOUT_PROGRESS_WORKER_INDEX=0 \
bash jl_patch/run_qwen3_vl_2b_grpo.sh
```

## 3. 环境变量说明

- `VERL_PROGRESS_LOG`
说明：总开关。`1` 开启详细进度日志；`0` 关闭。
默认：`1`（在 `jl_patch/run_qwen3_vl_2b_grpo.sh` 里设置）。

- `VERL_PROGRESS_LOG_INTERVAL`
说明：按 `global_step` 控制打印频率。仅在 `step % interval == 0` 时打印 progress。
默认：`1`。

- `VERL_PROGRESS_BAR_INTERVAL`
说明：actor 更新进度条刷新间隔（按 mini-step）。
默认：`1`。

- `VERL_PROGRESS_HEARTBEAT_SEC`
说明：长耗时阻塞阶段（当前用于 rollout gen）心跳间隔秒数。
默认：`5`。

- `VERL_PROGRESS_LOG_META_PREVIEW`
说明：driver 侧打印 meta keys 时的预览 key 数量上限。
默认：`6`（代码默认值，脚本未显式导出）。

- `VERL_ROLLOUT_PROGRESS_INTERVAL`
说明：rollout 进度条刷新间隔（按已完成 sample 数）。
默认：`1`。

- `VERL_ROLLOUT_PROGRESS_WORKER_INDEX`
说明：细粒度 rollout sample 进度由哪个 agent-loop worker 打印。
取值：`0` 仅 worker0；`-1` 全部 worker。
默认：`0`。

- `VERL_ROLLOUT_PROGRESS_BAR_WIDTH`
说明：rollout 进度条宽度。
默认：`24`（代码默认值，脚本未显式导出）。

## 4. 日志输出示例

driver 阶段日志：

```text
[VERL_PROGRESS][step=8] rollout start: prompts_shape=(128, 1024), rollout_n=4, meta_keys=['global_steps', 'temperature']
[VERL_PROGRESS][step=8] rollout gen running: elapsed=25.0s
[VERL_PROGRESS][step=8] reward done: reward_time=0.01s, reward_extra_keys=['reward']
[VERL_PROGRESS][step=8] update_actor start
[VERL_PROGRESS][step=8] update_actor done: 101.23s
```

actor mini-step 进度条：

```text
[VERL_PROGRESS][actor=dp][phase=update_policy] [##########------------] 6/12 (epoch 2/3, mini 2/4) elapsed=40.1s eta=40.1s
[VERL_PROGRESS][actor=megatron][phase=update_policy] [##############--------] 14/20 elapsed=92.4s eta=39.6s
```

rollout 实时进度条：

```text
[VERL_PROGRESS][step=8][rollout_manager][train] [########------------] 128/512 (25.0%) elapsed=22.4s eta=67.2s
[VERL_PROGRESS][step=8][rollout_worker=0][train] [############--------] 48/64 (75.0%) elapsed=19.1s eta=6.4s
```

## 5. 改动文件与改动点

- `verl/trainer/ppo/ray_trainer.py`
改动点：
- 新增 driver 侧 progress 控制与打印方法（`_should_log_progress`、`_progress_log`、`_summarize_dict_keys`）。
- 新增阶段心跳（`_start_phase_heartbeat`/`_stop_phase_heartbeat`）。
- 在 train/validate 主流程加入阶段 start/done 日志。

- `verl/workers/fsdp_workers.py`
改动点：
- 新增 rank0-only `_progress_log`。
- 在 `update_actor`、`generate_sequences` 增加 start/done 日志。

- `verl/workers/megatron_workers.py`
改动点：
- 新增 rank0-only `_progress_log`。
- 在 `update_actor`、`generate_sequences` 增加 start/done 日志。
- 在 `update_actor` 计算并传递 `total_iterations` 给 actor，用于进度条总数。

- `verl/workers/actor/dp_actor.py`
改动点：
- `update_policy` 增加 mini-step 级进度条（含 elapsed/eta）。

- `verl/workers/actor/megatron_actor.py`
改动点：
- `update_policy` 增加 mini-step 级进度条（含 elapsed/eta）。
- 支持 `total_iterations` 参数，已完成/总数可计算时显示条形进度。

- `verl/experimental/agent_loop/agent_loop.py`
改动点：
- 新增 rollout 进度条格式化。
- `AgentLoopWorker.generate_sequences` 由 `asyncio.gather` 改为 `asyncio.as_completed` 处理完成事件，实现 sample 级实时进度。
- `AgentLoopManager.generate_sequences` 用 `ray.wait` 逐 chunk 收集，实现全局样本完成进度。
- 增加“仅指定 worker 打细粒度进度”的控制。

- `jl_patch/run_qwen3_vl_2b_grpo.sh`
改动点：
- 导出 progress 相关环境变量，便于直接调节日志频率与 rollout 进度行为。

## 6. 多机多卡建议

- 推荐只开一个 worker 的细粒度 rollout：`VERL_ROLLOUT_PROGRESS_WORKER_INDEX=0`。
- 推荐把 `VERL_ROLLOUT_PROGRESS_INTERVAL` 调大到 `8/16/32`。
- 推荐按 step 降频：`VERL_PROGRESS_LOG_INTERVAL=2` 或更大。
- 若训练吞吐受影响，先关细粒度：`VERL_PROGRESS_LOG=0` 或增大各 interval。

## 7. 已知边界

- rollout 的“实时 completed/total”目前实现于 async agent-loop 路径。
- 非 async rollout 路径仍以阶段级日志与心跳为主，不提供同粒度 sample 完成计数。
- 打印本身会有轻微 CPU/IO 开销，worker 数越多越明显。

## 8. 文档维护约定

后续此功能若有任何修改，请同步更新本文件中的以下部分：

- `3. 环境变量说明`
- `5. 改动文件与改动点`
- `7. 已知边界`

建议在文档顶部追加一行更新时间。
