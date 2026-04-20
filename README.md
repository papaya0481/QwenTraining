# 安装

运行以下命令 以克隆包含 `ms-swift` 子模块的仓库.
```
git clone --recurse-submodules https://github.com/papaya0481/QwenTraining.git
```

## Live Code Bench

推荐使用uv作为管理器。安装
```
pip install uv
```
### 安装
进入 `LiveCodeBench` 目录，安装依赖。
```cd LiveCodeBench
uv pip install -e .
```
### 注意事项
需要安装 `datasets==3.6.0`[issue](https://github.com/LiveCodeBench/LiveCodeBench/issues/107)
```
uv pip install datasets==3.6.0
```

如果遇到错
```
 Failed to import vLLM: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /data/wuli_error/miniconda3/envs/llmqw/lib/python3.11/lib-dynload/../.././libicui18n.so.78)
 ```
需要添加：
```
export LD_LIBRARY_PATH=/data/wuli_error/miniconda3/envs/llmqw/lib:$LD_LIBRARY_PATH
```

### 计算
使用`compute.sh`获得不同级的分数。

## MS-enclave sandbox
>[!WARNING]
>MS-enclave sandbox 需要sudo权限以运行docker.
### 安装
```
uv pip install 'ms-enclave[docker]'
```
## MS-Swift 
### 安装
```
cd ms-swift
uv pip install -e .
```
安装vllm, deepspeed.
```
uv pip install vllm deepspeed --torch-backend=auto
```

## Verl
### VeRPO论文文件
在 `docs/acl_latex.tex` 中有 VeRPO 论文的 LaTeX 文件，供AI生成codes时参考。
### 安装flash attention
使用 [github proxy](https://gh-proxy.com)，去到[flash attention的release页面](https://github.com/Dao-AILab/flash-attention/releases)，下载对应pytorch, python, cuda版本的whl文件，安装。额外的编译好的安装文件[[1]](https://github.com/alkemiik-coder/FlashAttention-2.8.3-Custom-Linux-Wheels)[[2]](https://github.com/lesj0610/flash-attention/releases)
### 安装pyext
```
uv pip install git+https://github.com/ShaohonChen/PyExt.git@py311support
```
### 开启训练
开启ray
```
ray start --head --dashboard-port=8900
```
然后运行 `scripts/dapo/run_dapo.sh`.

### `gen_batch_size` 与 `train_batch_size` 的含义
这两个字段都表示“prompt 数量”，不是 `per_device_train_batch_size`，也不是单卡 batch size。

- `train_batch_size`
  表示一次 PPO/DAPO 训练更新希望使用的全局 prompt 数。它是跨所有 GPU 的总数，不是每张卡各自的数量。
- `gen_batch_size`
  表示 dataloader 每次先取出多少个 prompt 用来做 rollout / 采样。在当前 Ray trainer 实现中，如果显式设置了 `gen_batch_size`，dataloader 实际使用的是它；如果没有设置，则回退到 `train_batch_size`。

可以把两者理解为：

- `train_batch_size`：目标训练批大小。
- `gen_batch_size`：单次生成阶段先拿多少个 prompt 来采样。

常见关系如下：

1. 当 `gen_batch_size == train_batch_size`
   这是最直观的情况。每个训练 step 取出 `train_batch_size` 个 prompt，生成后直接进入训练。

2. 当 `gen_batch_size > train_batch_size`
   这通常用于 DAPO 的动态过滤场景。系统会先生成更大的候选 batch，再从中筛出足够的有效 group，最后凑满 `train_batch_size` 再训练。
   但需要注意：如果关闭了 `algorithm.filter_groups.enable`，当前自定义 trainer 实际会直接按 `gen_batch_size` 进入训练，因此这时“真实每 step 的 prompt 数”更接近 `gen_batch_size`。

3. 当 `gen_batch_size < train_batch_size`
   这种配置通常意味着一次生成不够，需要多次生成再拼接，常见于启用了 group filtering 的流程。否则容易让配置语义变得不直观。

和多卡的关系：

- 这两个值都是全局 batch size，不是单卡 batch size。
- 例如 `n_gpus=2`、`train_batch_size=2`，含义是“一次训练更新总共使用 2 个 prompt”，不是“每张卡 2 个 prompt”。
- 如果负载均匀，两张卡大致各处理 1 个 prompt 对应的数据。

和 `rollout.n` 的关系：

- 若 `rollout.n = 8`，则每个 prompt 会生成 8 条 response / trajectory。
- 因此一次 step 的 trajectory 数约为：
  `prompt_batch_size * rollout.n`
- 其中 `prompt_batch_size` 在理想语义上应是 `train_batch_size`；但如果当前实现直接按 dataloader batch 训练，则会实际变成 `gen_batch_size`。

如何估算一个 epoch 有多少 step：

- 若当前训练实际每 step 使用 `B` 个 prompt，则：
  `epoch_steps = 数据集样本数 / B`
- 例如 9000 条数据：
  - 若真实每 step 用 2 个 prompt，则一轮约 `9000 / 2 = 4500` step。
  - 若真实每 step 用 4 个 prompt，则一轮约 `9000 / 4 = 2250` step。

对于本仓库当前的 `scripts/dapo/config/dapo_qwen3_5_0_8b.yaml`：

- 配置写的是 `gen_batch_size: 4`
- 配置写的是 `train_batch_size: 2`
- 且 `filter_groups.enable: false`

因此按当前实现，实际 step 数更接近按 `gen_batch_size=4` 计算，而不是按 `train_batch_size=2` 计算。

### `compute_score` 返回指标说明
`scripts/plugins/verl_codegen1_reward.py` 中的 `compute_score` 会返回一个字典。各字段含义如下：

| Key | 含义 |
| --- | --- |
| `score` | 当前训练实际使用的最终 reward。当前实现里，它等于 `traj_reward`。 |
| `acc` | 当前样本的测试通过率，范围是 `[0, 1]`。 |
| `passed` | 当前样本通过的测试用例数。 |
| `total` | 当前样本参与评测的测试用例总数。 |
| `pass_rate` | 与 `acc` 相同，等于 `passed / total`。 |
| `dense_reward` | 基于每个测试用例结果和难度权重计算出的 dense shaping reward。 |
| `traj_reward` | `outcome_reward` 再乘上效率衰减后的结果。 |
| `mixed_reward_preview` | `dense_reward` 和 `traj_reward` 的加权混合预览值，仅用于观察或调试。 |
| `outcome_reward` | 原始任务结果奖励。当前规则下，全部测试通过时为 `1.0`，否则为 `0.0`。 |
| `efficiency_decay` | 效率惩罚系数。轨迹越长，例如 turn 更多或 token 更多，这个值通常越小。 |
| `avg_difficulty_weight` | 计算 `dense_reward` 时使用的平均测试用例难度权重。 |
| `density_sigma` | dense reward 公式中的 spread 统计量，主要用于分析和调试。 |

### 训练日志指标说明

以下说明对应当前 DAPO / PPO 训练日志中常见的指标，适用于类似下面这种按 step 打印的日志：

```text
step:2 - global_seqlen/min:... - actor/entropy:... - timing_s/gen:... - perf/throughput:...
```

#### 命名约定

- `/mean`、`/max`、`/min`：分别表示均值、最大值、最小值。
- `/clip_ratio`：命中长度上限的比例。
- `timing_s/*`：某阶段耗时，单位是秒。
- `timing_per_token_ms/*`：某阶段的单 token 耗时，单位是毫秒。
- `timing_ratio/rollout_in_step`：rollout 生成时间占整个训练 step 的比例，等于 `timing_s/gen / timing_s/step`。
- `val-core/.../mean@1`、`val-aux/.../mean@1`：验证集上的 top-1 聚合指标。

#### Step 与 batch 级指标

| Metric | 含义 |
| --- | --- |
| `step` | 当前训练步数。 |
| `train/num_gen_batches` | 为凑齐当前训练 step，实际做了多少次 generation batch。开启 group filtering 时，这个值可能大于 1。 |

#### 序列长度与 batch 平衡指标

| Metric | 含义 |
| --- | --- |
| `global_seqlen/min` | 当前 step 中，各 DP rank 原始总序列长度的最小值。 |
| `global_seqlen/max` | 当前 step 中，各 DP rank 原始总序列长度的最大值。 |
| `global_seqlen/minmax_diff` | `global_seqlen/max - global_seqlen/min`，用于观察负载是否不均衡。 |
| `global_seqlen/balanced_min` | 做完 batch balance 后，各 DP rank 总序列长度的最小值。 |
| `global_seqlen/balanced_max` | 做完 batch balance 后，各 DP rank 总序列长度的最大值。 |
| `global_seqlen/mean` | 当前 step 各 rank 总序列长度的平均值。 |

#### Actor 相关指标

| Metric | 含义 |
| --- | --- |
| `actor/entropy` | 当前 batch 上 actor token 分布的平均熵，反映策略输出的不确定性。 |
| `actor/pg_clipfrac` | PPO 中 advantage 为正时，被 clip 的 token 比例。 |
| `actor/pg_clipfrac_lower` | PPO 中 advantage 为负时，被 clip 的 token 比例。 |
| `actor/ppo_kl` | 当前 actor 与 old / ref policy 的近似 KL 散度。 |
| `actor/pg_loss` | policy gradient 部分的 loss。 |
| `actor/loss` | actor 最终优化使用的 loss。通常等于或接近 `actor/pg_loss`，若有额外正则项则会包含进去。 |
| `actor/grad_norm` | actor 更新时的梯度范数。 |
| `actor/lr` | 当前 actor 学习率。 |

#### Critic / reward / advantage 指标

| Metric | 含义 |
| --- | --- |
| `critic/score/mean, /max, /min` | 每条轨迹最终 `token_level_scores` 求和后的统计量。通常是 reward function 的直接输出。 |
| `critic/rewards/mean, /max, /min` | 每条轨迹最终 `token_level_rewards` 求和后的统计量。若启用了 KL-in-reward，它会和 `score` 不同。 |
| `reward_extra/dense_reward/mean, /max, /min` | reward 函数额外返回的 dense reward 统计量。当前 DAPO / VeRPO 配置下，它是 testcase 级 shaping reward：先按同一 prompt 下整组 rollout 的 testcase 通过情况得到组级权重，再对当前样本的 testcase 通过标记做加权求和，形式上可理解为 `dense_reward = Σ_j(w'_j * q_j)`，其中 `q_j` 是第 `j` 个 testcase 是否通过，`w'_j` 是结合 testcase 难度和组内通过率密度后的归一化权重。 |
| `reward_extra/traj_reward/mean, /max, /min` | reward 函数额外返回的 trajectory reward 统计量。当前实现里它直接由结果奖励乘效率衰减得到，即 `traj_reward = outcome_reward * efficiency_decay`。其中 `outcome_reward` 在全部测试通过时为 `1.0`，否则为 `0.0`；`efficiency_decay` 会根据轨迹效率惩罚，一般按 turn 数或 response token 数做衰减。 |
| `critic/advantages/mean, /max, /min` | 有效 response token 上 advantage 的统计量。 |
| `critic/returns/mean, /max, /min` | 有效 response token 上 return 的统计量。 |
| `critic/values/mean, /max, /min` | critic 预测 value 的统计量。只在启用 critic 时出现。 |
| `critic/vf_explained_var` | value function explained variance，越高通常表示 critic 拟合 return 越好。 |

#### Prompt / response / 轨迹结构指标

| Metric | 含义 |
| --- | --- |
| `response_length/mean, /max, /min` | response 长度统计，包含 aborted 样本。 |
| `response_length/clip_ratio` | response 长度命中 `max_response_length` 的比例。 |
| `response_length_non_aborted/mean, /max, /min` | 非 aborted 样本上的 response 长度统计。 |
| `response_length_non_aborted/clip_ratio` | 非 aborted 样本中命中长度上限的比例。 |
| `response/aborted_ratio` | response 长度为 0 的样本比例。 |
| `prompt_length/mean, /max, /min` | prompt 长度统计。 |
| `prompt_length/clip_ratio` | prompt 长度命中 `max_prompt_length` 的比例。 |
| `num_turns/min, /max, /mean` | 多轮对话中 turn 数的统计量。 |

#### Timing 指标

| Metric | 含义 |
| --- | --- |
| `timing_s/start_profile` | profiler 启动逻辑耗时。 |
| `timing_s/gen` | rollout 生成阶段总耗时。通常可以把它视为 rollout 主耗时。 |
| `timing_s/reward` | reward 计算耗时。 |
| `timing_s/old_log_prob` | 重新计算 old log prob 的耗时。 |
| `timing_s/ref` | 参考策略 log prob 计算耗时。启用 reference policy 时出现。 |
| `timing_s/values` | critic value 前向耗时。启用 critic 时出现。 |
| `timing_s/adv` | advantage 计算耗时。 |
| `timing_s/update_critic` | critic 参数更新耗时。 |
| `timing_s/update_actor` | actor 参数更新耗时。 |
| `timing_s/update_weights` | 将 actor 最新权重同步给 rollout / ref 等组件的耗时。 |
| `timing_s/save_checkpoint` | 保存 checkpoint 耗时。发生在触发保存的 step。 |
| `timing_s/testing` | validation / testing 耗时。发生在触发验证的 step。 |
| `timing_s/stop_profile` | profiler 关闭逻辑耗时。 |
| `timing_s/step` | 整个训练 step 的总耗时。 |
| `timing_s/agent_loop/num_preempted/min, /max, /mean` | agent loop 中被抢占的次数统计。没有启用相关机制时可能是 `-1`。 |
| `timing_s/agent_loop/generate_sequences/min, /max, /mean` | 单条样本在 agent loop 内生成序列的耗时统计。 |
| `timing_s/agent_loop/tool_calls/min, /max, /mean` | 单条样本在 agent loop 内 tool call 的耗时统计。 |
| `timing_s/agent_loop/slowest/generate_sequences` | 最慢样本的序列生成耗时。 |
| `timing_s/agent_loop/slowest/tool_calls` | 最慢样本的 tool call 耗时。 |
| `timing_s/agent_loop/slowest/num_preempted` | 最慢样本对应的抢占次数。 |
| `timing_s/agent_loop/slowest/prompt_length` | 最慢样本对应的 prompt 长度。 |
| `timing_s/agent_loop/slowest/response_length` | 最慢样本对应的 response 长度。 |
| `timing_per_token_ms/gen` | rollout 生成阶段平均每个 response token 的耗时。 |
| `timing_per_token_ms/adv` | advantage 计算的单 token 耗时。 |
| `timing_per_token_ms/update_actor` | actor 更新阶段的单 token 耗时。 |
| `timing_per_token_ms/ref` | ref log prob 计算的单 token 耗时。启用 reference policy 时出现。 |
| `timing_per_token_ms/values` | critic value 前向的单 token 耗时。启用 critic 时出现。 |
| `timing_per_token_ms/update_critic` | critic 更新的单 token 耗时。启用 critic 时出现。 |
| `timing_ratio/rollout_in_step` | rollout 生成耗时占整个 step 的比率，值越高说明 step 更受 rollout 侧限制。 |

#### 性能指标

| Metric | 含义 |
| --- | --- |
| `perf/mfu/actor_infer` | actor 推理阶段的 MFU（Model FLOPs Utilization）估计。 |
| `perf/mfu/actor` | actor 训练更新阶段的 MFU 估计。 |
| `perf/total_num_tokens` | 当前 step 总 token 数，通常是所有 prompt + response token 的总和。 |
| `perf/time_per_step` | 当前 step 总耗时，和 `timing_s/step` 对应。 |
| `perf/throughput` | 平均每秒每张 GPU 处理的 token 数。 |

#### Validation 指标

下面这类指标通常出现在触发 `test_freq` 的 step。

| Metric | 含义 |
| --- | --- |
| `val-core/taco_rl/acc/mean@1` | 验证集 top-1 通过率 / 准确率，是最核心的主指标。 |
| `val-aux/taco_rl/reward/mean@1` | 验证集 top-1 最终 reward 均值。 |
| `val-aux/taco_rl/score/mean@1` | 验证集 top-1 `score` 均值。 |
| `val-aux/taco_rl/passed/mean@1` | 验证集中平均通过的测试用例数。 |
| `val-aux/taco_rl/total/mean@1` | 验证集中平均测试用例总数。 |
| `val-aux/taco_rl/pass_rate/mean@1` | 验证集平均通过率，通常和 `acc` 同义或非常接近。 |
| `val-aux/taco_rl/dense_reward/mean@1` | 验证集平均 dense reward。 |
| `val-aux/taco_rl/traj_reward/mean@1` | 验证集平均 trajectory reward。 |
| `val-aux/taco_rl/mixed_reward_preview/mean@1` | 验证集平均 mixed reward 预览值，仅用于观察。 |
| `val-aux/taco_rl/outcome_reward/mean@1` | 验证集平均 outcome reward。 |
| `val-aux/taco_rl/efficiency_decay/mean@1` | 验证集平均效率衰减系数。 |
| `val-aux/taco_rl/avg_difficulty_weight/mean@1` | 验证集平均难度权重。 |
| `val-aux/taco_rl/density_sigma/mean@1` | 验证集平均 density sigma。 |
| `val-aux/num_turns/min, /max, /mean` | 验证样本的 turn 数统计。 |

上面 `val-aux/taco_rl/*` 的奖励相关字段，与前面 `compute_score` 返回指标说明是一一对应的，只是这里是对整个验证集做聚合后的日志输出。


## 其他
1. 关于多卡运行的问题[issue](https://github.com/modelscope/ms-swift/issues/3991)
2. Qwen3.5的最佳时实践[Qwen3.5](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html)
