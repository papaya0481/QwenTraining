# Reward 函数说明

本文只讨论当前项目训练阶段的 reward 设计与实现，不讨论 `LiveCodeBench` benchmark，也不讨论 `data/` 数据构造。

---

## 1. reward 在这个项目里的位置

当前项目的 RL 训练主要走：

- `scripts/dapo/main_dapo.py`
- `scripts/dapo/dapo_ray_trainer.py`
- `verl` 的 PPO / DAPO / reward loop

reward 的职责是：

- 对 rollout 生成的代码结果做打分
- 把执行反馈转成训练信号
- 最终供 advantage 计算和 actor 更新使用

这里的 reward 是训练信号，不是 benchmark 分数。

---

## 2. 当前仓库的 reward 相关文件

最核心的文件有：

- 自定义 reward 函数
  [scripts/plugins/verl_codegen1_reward.py](/home/ruixin/workspace/QwenTraining/scripts/plugins/verl_codegen1_reward.py)
- 自定义 VeRPO reward manager
  [scripts/plugins/verpo_reward_manager.py](/home/ruixin/workspace/QwenTraining/scripts/plugins/verpo_reward_manager.py)
- 实验 reward loop 中的 VeRPO manager
  [verl/verl/experimental/reward_loop/reward_manager/verpo.py](/home/ruixin/workspace/QwenTraining/verl/verl/experimental/reward_loop/reward_manager/verpo.py)
- 标准 DAPO reward manager
  [verl/verl/workers/reward_manager/dapo.py](/home/ruixin/workspace/QwenTraining/verl/verl/workers/reward_manager/dapo.py)
- reward 装载逻辑
  [verl/verl/trainer/ppo/reward.py](/home/ruixin/workspace/QwenTraining/verl/verl/trainer/ppo/reward.py)
- 训练入口
  [scripts/dapo/main_dapo.py](/home/ruixin/workspace/QwenTraining/scripts/dapo/main_dapo.py)
- 自定义 trainer
  [scripts/dapo/dapo_ray_trainer.py](/home/ruixin/workspace/QwenTraining/scripts/dapo/dapo_ray_trainer.py)
- 执行器
  [data/code_excutor.py](/home/ruixin/workspace/QwenTraining/data/code_excutor.py)

---

## 3. 高层流程

训练时 reward 的主路径可以概括为：

1. rollout 生成代码
2. reward manager 收到生成结果
3. reward manager 调执行器跑测试
4. 根据 testcase 结果生成 `score` / `dense_reward` / `traj_reward`
5. trainer 将这些结果写入 batch
6. advantage 计算使用这些字段

这个项目最大的特点是：

- reward 基本是“执行反馈驱动”
- 不是单独训练一个黑盒 reward model 再给分

---

## 4. `compute_score` 做了什么

最核心的自定义 reward 函数在：

- [scripts/plugins/verl_codegen1_reward.py](/home/ruixin/workspace/QwenTraining/scripts/plugins/verl_codegen1_reward.py)

它里面有两个重要层次。

### 第一层：`execute_single`

`execute_single(...)` 做的事情是：

1. 解析 `ground_truth` 中的测试样例
2. 根据 `fn_mode` 决定执行方式
3. 调 `ModelResponseCodeExecutor.evaluate(...)`
4. 汇总 per-testcase 通过情况

它最终产出的是比较原始的执行统计：

- `passed_flags`
- `passed`
- `total`
- `pass_rate`
- `outcome_reward`
- `turn_count`

这里的 `outcome_reward` 当前语义非常明确：

- 全部 testcase 通过，则为 `1.0`
- 否则为 `0.0`

### 第二层：`compute_score`

`compute_score(...)` 是单样本入口，主要用于：

- 单样本 reward 路径
- fallback 路径
- 验证路径

它会在 `execute_single` 的输出上进一步构造：

- `dense_reward`
- `traj_reward`
- `mixed_reward_preview`
- `score`

当前实现里最关键的一点是：

- 最终训练使用的 `score` 仍然等于 `traj_reward`

也就是说，虽然脚本里计算了 `dense_reward`，但它不是直接拿来当最终 reward tensor 的唯一值。

---

## 5. `dense_reward`、`traj_reward`、`score` 分别是什么

### `dense_reward`

`dense_reward` 表示基于 testcase 局部通过情况构造的 dense shaping reward。

它不是简单的 `pass_rate`，而是：

1. 先根据组内 testcase 通过率计算难度权重
2. 再用 density normalization 压制“全是简单 testcase”带来的冗余奖励
3. 最后把当前样本通过的 testcase 权重加总

这个逻辑对应 VeRPO 的核心思想：

- 不是只看最终全对/全错
- 也不是只看朴素通过率
- 而是看“带权局部成功”

### `traj_reward`

`traj_reward` 表示轨迹级奖励。

当前实现里它大致是：

- `outcome_reward * efficiency_decay`

也就是说：

- 最终正确性是主锚点
- 再乘一个效率衰减项

如果 `efficiency_mode = turn_count`，那么：

- 轨迹越长
- turn 越多
- 衰减越明显

### `score`

当前实现里：

- `score = traj_reward`

因此：

- `score` 才是训练实际直接使用的最终标量 reward
- `dense_reward` 更像额外提供给 VeRPO advantage 的局部信号

---

## 6. VeRPO 的关键不是单样本打分，而是组内打分

如果只看 `compute_score`，容易误以为 reward 只是“单样本执行 + 单样本打分”。

但 VeRPO 真正关键的部分，是 group-level reward。

对应实现有两套：

- [scripts/plugins/verpo_reward_manager.py](/home/ruixin/workspace/QwenTraining/scripts/plugins/verpo_reward_manager.py)
- [verl/verl/experimental/reward_loop/reward_manager/verpo.py](/home/ruixin/workspace/QwenTraining/verl/verl/experimental/reward_loop/reward_manager/verpo.py)

它们的核心逻辑都是：

1. 同一个 prompt 先采样出 N 条 response
2. 对这 N 条 response 全部执行测试
3. 统计组内每个 testcase 的通过率 `rho_j`
4. 用组内 `rho_j` 计算 testcase 难度权重
5. 再给组内每条样本算 `dense_reward`

这和普通 DAPO 的本质区别是：

- DAPO 更容易逐条样本独立打分
- VeRPO 强依赖“组内相对统计量”

因此 VeRPO 的 `dense_reward` 不是静态标签，而是在线 rollout group 上下文相关的。

---

## 7. reward manager 在整条链里的作用

reward manager 的职责不是“发明 reward”，而是：

- 组织 batch / group
- 调 reward function
- 返回 `reward_tensor` 和 `reward_extra_info`

在标准 DAPO 路径里：

- [verl/verl/workers/reward_manager/dapo.py](/home/ruixin/workspace/QwenTraining/verl/verl/workers/reward_manager/dapo.py)

更像逐样本调用 `compute_score(...)`。

在 VeRPO 路径里：

- `run_batch(...)`

会优先被 reward loop 调用，用来保留组内统计信息。

这点在训练效果上很关键，因为一旦退化回逐样本路径：

- `rho_j`
- difficulty weight
- density normalization

就都不再是论文意义上的 group-level 统计了。

---

## 8. trainer 如何消费 reward

在：

- [scripts/dapo/dapo_ray_trainer.py](/home/ruixin/workspace/QwenTraining/scripts/dapo/dapo_ray_trainer.py)

训练 loop 中，reward 的主要消费方式是：

1. 调 reward loop 得到 `rm_scores`
2. 用 `extract_reward(...)` 取出 `reward_tensor` 和 `reward_extra_infos_dict`
3. 写入 `token_level_scores`
4. 若不开 KL-in-reward，则直接把 `token_level_rewards = token_level_scores`
5. 再进入 `compute_advantage(...)`

所以从 trainer 视角看：

- `reward_tensor` 是训练主输入
- `reward_extra_infos_dict` 是辅助统计与 VeRPO advantage 额外通道

---

## 9. VeRPO advantage 如何使用这些 reward

VeRPO advantage 实现在：

- [verl/verl/trainer/ppo/core_algos.py](/home/ruixin/workspace/QwenTraining/verl/verl/trainer/ppo/core_algos.py)

对应函数：

- `compute_verpo_advantage(...)`

它会优先读取：

- `traj_reward`
- `dense_reward`

然后构造：

- 轨迹级优势 `A_traj`
- turn 级 dense 优势 `A_turn`

最后按：

- `combined = A_traj + beta * A_turn`

组合成最终 advantage。

所以项目当前的结构可以总结为：

- `traj_reward` 负责全局正确性锚点
- `dense_reward` 负责局部 dense 信号
- `score` 是当前单样本最终 reward 标量
- VeRPO advantage 再把 `traj_reward` 和 `dense_reward` 融合

---

## 10. 当前配置里最需要注意的地方

以：

- [scripts/dapo/config/dapo_qwen3_5_0_8b.yaml](/home/ruixin/workspace/QwenTraining/scripts/dapo/config/dapo_qwen3_5_0_8b.yaml)

为例，关键配置是：

- `algorithm.adv_estimator: verpo`
- `reward.reward_manager.name: verpo`
- `reward.custom_reward_function.path: scripts/plugins/verl_codegen1_reward.py`

这三者的语义分别是：

- `adv_estimator: verpo`
  表示 advantage 用 VeRPO 公式算
- `reward_manager.name: verpo`
  表示 reward manager 使用 group-aware VeRPO 逻辑
- `custom_reward_function.path`
  提供底层单样本执行与 reward 计算函数

这三者不是一回事，不能混为“只改了一个名字就全部切换”。

---

## 11. 当前实现的强项

### 1. reward grounded in execution

当前 reward 直接来自 testcase 执行反馈，可靠性通常比黑盒 RM 更高。

### 2. group-aware dense reward

VeRPO 利用组内通过率构造难度权重，比朴素通过率更细。

### 3. `reward_extra_info` 丰富

当前会保留：

- `pass_rate`
- `dense_reward`
- `traj_reward`
- `avg_difficulty_weight`
- `density_sigma`

这让后续分析更方便。

---

## 12. 当前实现的边界

### 1. `score` 目前仍偏向 outcome-anchored

虽然计算了 `dense_reward`，但当前单样本路径的最终 `score` 仍然等于 `traj_reward`。

### 2. reward manager 路径和配置表面语义不完全一致

特别是 VeRPO 场景下，真正重要的是 `run_batch(...)` 有没有被调用，而不只是 YAML 里是否写了 `compute_score`。

### 3. reward 不是 benchmark 分数

训练时 reward 和最终 LiveCodeBench 的 `pass@1` 是两个层次的指标，不能直接混用。

---

## 13. 文件索引

- reward 函数
  [scripts/plugins/verl_codegen1_reward.py](/home/ruixin/workspace/QwenTraining/scripts/plugins/verl_codegen1_reward.py)
- VeRPO reward manager
  [scripts/plugins/verpo_reward_manager.py](/home/ruixin/workspace/QwenTraining/scripts/plugins/verpo_reward_manager.py)
- 实验 reward loop manager
  [verl/verl/experimental/reward_loop/reward_manager/verpo.py](/home/ruixin/workspace/QwenTraining/verl/verl/experimental/reward_loop/reward_manager/verpo.py)
- 标准 DAPO reward manager
  [verl/verl/workers/reward_manager/dapo.py](/home/ruixin/workspace/QwenTraining/verl/verl/workers/reward_manager/dapo.py)
- reward 装载
  [verl/verl/trainer/ppo/reward.py](/home/ruixin/workspace/QwenTraining/verl/verl/trainer/ppo/reward.py)
- trainer
  [scripts/dapo/dapo_ray_trainer.py](/home/ruixin/workspace/QwenTraining/scripts/dapo/dapo_ray_trainer.py)
- advantage
  [verl/verl/trainer/ppo/core_algos.py](/home/ruixin/workspace/QwenTraining/verl/verl/trainer/ppo/core_algos.py)
