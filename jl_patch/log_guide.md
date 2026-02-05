# verl PPO训练日志解读指南

## 日志位置
```bash
jl_patch/verl_training.log
```

## 常用查看命令

### 1. 实时查看训练日志
```bash
tail -f jl_patch/verl_training.log
```

### 2. 查看最近的训练步骤
```bash
tail -100 jl_patch/verl_training.log | grep "^step:"
```

### 3. 查看训练进度
```bash
grep "Training Progress" jl_patch/verl_training.log | tail -5
```

### 4. 查看特定step的详细信息
```bash
grep "step:10 " jl_patch/verl_training.log
```

### 5. 查看evaluation结果（每10步）
```bash
grep "val/test_score" jl_patch/verl_training.log
```

## 关键指标解读

每个step的日志包含很多指标，以下是最重要的：

### 训练进度
- `training/global_step`: 当前全局步数
- `training/epoch`: 当前epoch
- `Training Progress: X%`: 整体进度百分比

### 性能指标
- `critic/score/mean`: **核心指标** - 平均得分（GSM8K准确率的代理指标）
- `critic/rewards/mean`: 平均reward（1.0=答对，0.0=答错）
- `response_length/mean`: 平均回答长度
- `perf/throughput`: 每秒处理的tokens数

### 训练损失
- `actor/pg_loss`: Actor的policy gradient损失
- `actor/entropy`: 策略熵（越高越探索）
- `critic/vf_loss`: Critic的value function损失
- `actor/ppo_kl`: KL散度（与reference policy的差异）

### 时间分析
- `timing_s/gen`: 生成阶段耗时（秒）
- `timing_s/update_critic`: 更新Critic耗时
- `timing_s/update_actor`: 更新Actor耗时
- `timing_s/step`: 总耗时

### 梯度信息
- `actor/grad_norm`: Actor梯度范数
- `critic/grad_norm`: Critic梯度范数

### Evaluation指标（每10步）
- `val/test_score/openai/gsm8k`: **最重要** - GSM8K测试集准确率
- 这是模型在测试集上的真实表现

## 示例：理解一行日志

```
step:5 - training/global_step:5 - critic/score/mean:0.0234375 -
critic/rewards/mean:0.0234375 - timing_s/step:41.01343672798248
```

**解读：**
- 第5步训练
- 平均得分：2.34%（256个样本中约6个答对）
- 本step耗时：41秒

## 监控训练健康度

### 正常训练的信号
- ✅ `critic/score/mean` 逐步上升
- ✅ `actor/entropy` 在0.4-0.6之间（保持探索）
- ✅ `actor/ppo_kl` 接近0（不要偏离太多）
- ✅ `timing_s/step` 保持稳定（不要越来越慢）

### 需要注意的信号
- ⚠️ `critic/score/mean` 一直是0（模型没学到东西）
- ⚠️ `actor/grad_norm` 突然暴涨（梯度爆炸）
- ⚠️ `perf/max_memory_allocated_gb` 接近GPU显存上限（可能OOM）

## 快速查看训练曲线

### 提取score随时间变化
```bash
grep "^step:" jl_patch/verl_training.log | grep -o "critic/score/mean:[0-9.]*" | cut -d: -f2
```

### 提取每步耗时
```bash
grep "^step:" jl_patch/verl_training.log | grep -o "timing_s/step:[0-9.]*" | cut -d: -f2
```

### 查看最近10步的关键指标
```bash
grep "^step:" jl_patch/verl_training.log | tail -10 | \
  grep -o "training/global_step:[0-9]* \|critic/score/mean:[0-9.]* \|timing_s/step:[0-9.]*"
```

## WandB查看

如果配置了WandB，可以在网页端查看更直观的图表：
- 访问 https://wandb.ai
- 找到项目：`verl-gsm8k`
- 查看实验：`ppo-qwen2.5-0.5b-single-gpu`

在WandB中可以看到：
- 实时训练曲线
- Evaluation结果图表
- 系统资源使用情况
- 详细的超参数配置
