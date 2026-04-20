# DAPO vLLM LoRA 动态注入修复总结

本文总结一次发生在 `DAPO + VERL + vLLM rollout` 场景下的 LoRA 动态注入问题，重点说明：

- 现象是什么
- 为什么 `LiveCodeBench/run.sh` 可以加载，但 DAPO 会报错
- 最终确认的根因是什么
- 这次具体做了哪些修复
- 为什么目前判断注入链已经稳定

---

## 1. 问题背景

当前训练链路使用的是：

- actor 侧：`FSDP2`
- rollout 侧：`vLLM`
- LoRA 形式：从已有 `lora_adapter_path` 继续训练，并在训练过程中把最新 LoRA tensor 动态同步到 rollout 模型

对应配置见：

- `scripts/dapo/config/dapo_qwen3_5_0_8b.yaml`

问题最初表现为：

- actor 侧可以正常从本地 `lora_adapter_path` 加载 LoRA
- 但 rollout 侧在 `update_weights -> add_lora` 阶段报错
- 报错最终落在 vLLM 的 LoRA manager 内部，典型堆栈为：

```text
first_lora: LoRALayerWeights = next(iter(lora_model.loras.values()))
StopIteration
```

这说明 vLLM 收到了一个 LoRA 请求，但最终没有解析出任何可挂载的 LoRA layer。

---

## 2. 为什么这不是 LoRA 文件本身坏掉

一个容易混淆的点是：

- `LiveCodeBench/run.sh` 能正常加载同一份 LoRA
- DAPO 却会在 rollout 时报错

这两者并不矛盾，因为它们走的是两条不同的路径。

### 2.1 `LiveCodeBench/run.sh` 走的是本地 adapter 直加载

它本质上是：

- 直接把 `lora_path` 传给 vLLM
- 由 vLLM 自己从磁盘读取 `adapter_config.json` 和 `adapter_model.safetensors`

这条路径验证的是：

- LoRA 目录存在
- LoRA 文件格式正确
- vLLM 可以直接侧载这份 adapter

### 2.2 DAPO 走的是训练时 tensor 动态注入

DAPO 不是让 vLLM 再次从磁盘读取 adapter，而是：

1. actor 侧先通过 PEFT 加载 LoRA
2. 在 `update_weights` 时抽取当前训练步的 LoRA tensor
3. 通过 VERL 的 `TensorLoRARequest` 把 tensor 发送给 rollout/vLLM
4. 由 vLLM 在内存中解析这些 tensor 并挂载 LoRA

因此，`LiveCodeBench/run.sh` 能成功，只能说明“LoRA 文件本身没问题”，不能说明“训练时动态 tensor 注入链路也没问题”。

---

## 3. 早期定位结论

在正式修复前，先补了一轮定位日志，结论逐步收敛到以下几点。

### 3.1 不是 `broadcast_from_rank0=False` 导致 LoRA 丢失

在 `verl/verl/utils/fsdp_utils.py` 的 `layered_summon_lora_params(...)` 中，FSDP2 使用：

```python
StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False)
```

初看会怀疑：

- 多卡下是不是只有 rank0 有 LoRA，其他卡拿不到

但实际日志表明：

- actor 侧确实成功抽取到了 LoRA tensor
- 不是 FSDP 收集阶段把 LoRA 丢了

因此，这不是本次问题的根因。

### 3.2 actor 侧已经拿到了完整 LoRA tensor

早期日志显示：

- `tensor_keys=372`
- 传给 vLLM 前 `normalized_keys=372`

说明：

- actor 侧成功从 PEFT/FSDP 模型中收集到了 LoRA 权重
- LoRA tensor 在传输过程中也没有丢失

### 3.3 `default` 不是问题

原始 key 形如：

```text
...lora_A.default.weight
...lora_B.default.weight
```

在传入 vLLM 之前，代码已经做了规范化：

```text
...lora_A.weight
...lora_B.weight
```

因此：

- PEFT adapter 名里的 `default` 并不是导致失败的原因

### 3.4 真正的问题出在 vLLM 的 tensor 解析阶段

日志表明：

- `input_keys=372`
- `normalized_keys=372`
- 但早期 `parsed_layers=0`

这意味着：

- LoRA tensor 已经到了 vLLM
- 但 vLLM 没有把这些 tensor 解析成任何有效的 LoRA layer

换句话说，问题不在磁盘加载，而在：

- `TensorLoRARequest -> from_lora_tensors()` 这条动态注入链路

---

## 4. 本次修复做了什么

这次改动没有改训练算法逻辑，主要做了三类工作：

- 增强可观测性
- 修复 rollout 侧空 tensor 的异常路径
- 保守地兼容 Qwen3.5 LoRA tensor 注入

---

## 5. 修复点一：补齐关键定位日志

### 5.1 FSDP2 LoRA 收集日志

在：

- `verl/verl/utils/fsdp_utils.py`

的 `layered_summon_lora_params(...)` 中增加了日志，打印：

- 当前 rank
- `full_state` 键数量
- `lora_state` 键数量
- 前几个 LoRA key

这样可以快速判断：

- LoRA 是否已经在 actor 侧成功收集

### 5.2 rollout 注入前日志

在：

- `verl/verl/workers/rollout/vllm_rollout/utils.py`

的 `_update_weights(...)` 中增加了日志，打印：

- 当前 bucket 收到多少个 LoRA tensor
- 前几个 key

这样可以判断：

- rollout worker 当前这次是否真的收到了 LoRA tensor

### 5.3 vLLM tensor 解析日志

在：

- `verl/verl/utils/vllm/utils.py`

的 LoRA hijack 路径中增加了日志，打印：

- 原始输入 key 数量
- 规范化后 key 数量
- 样例 key
- `expected_lora_modules`
- `from_lora_tensors()` 解析出的 `parsed_layers`

这一步帮助最终确认：

- 问题发生在 vLLM 的 tensor 解析，而不是更早阶段

---

## 6. 修复点二：空 LoRA tensor 的 worker 直接跳过

后续日志表明，在当前 TP/rollout 拓扑下，不是每个 vLLM worker 都会收到 LoRA tensor：

- 一个 worker 持续收到 `tensor_keys=372`
- 另一个 worker 持续收到 `tensor_keys=0`

如果对 `tensor_keys=0` 的 worker 仍然强行执行 `add_lora`，就会触发无意义的异常路径。

因此在：

- `verl/verl/workers/rollout/vllm_rollout/utils.py`

中增加了保护逻辑：

- 如果当前 worker 收到的 LoRA tensor 为空，则记录日志并直接跳过 `add_lora`

这一步的目标不是“绕过错误”，而是避免把“本次没有收到任何 LoRA tensor”的正常分布式情况，误处理成 LoRA 加载失败。

---

## 7. 修复点三：保守处理 `from_lora_tensors()` 的注入路径

### 7.1 保留原始 LoRA 能力，不删 key，不做激进融合

中间曾尝试过参考 `ms-swift` 的融合思路，把某些 `in_proj_*` 模块拼接成更大的投影模块。

但随后日志显示，当前环境下 vLLM 的 `expected_lora_modules` 已经包含：

- `in_proj_qkv`
- `in_proj_z`
- `in_proj_b`
- `in_proj_a`
- `out_proj`

这说明当前 vLLM 已经支持这些拆开的模块名。

因此最终修复采用了更保守的策略：

- 不删除任何 LoRA key
- 不对 `in_proj_*` 做融合
- 不为了“跑通”而牺牲原始模块表达能力

### 7.2 对 PEFT adapter 名做最小规范化

保留的规范化只有一项：

- 把 `...lora_A.default.weight` 变成 `...lora_A.weight`
- 把 `...lora_B.default.weight` 变成 `...lora_B.weight`

这一步只是在去掉 adapter 名，不改变模块结构。

### 7.3 只保留少量前缀/别名候选路径

在：

- `verl/verl/utils/vllm/utils.py`

中增加了 `_build_qwen35_tensor_variants(...)`，用于尝试少量、保守的 key 变体：

- 原始 `normalized`
- 少量前缀重写候选
- `out_proj -> proj` 的别名候选

这些候选的目的不是强行改模型结构，而是处理不同后端之间可能存在的路径前缀差异。

最终从日志看，真正生效的是：

- `variant=normalized`

也就是说，在当前环境里，最原始的规范化结果就足够成功注入。

---

## 8. 当前修复后的稳定性结论

从连续多个训练 step 的日志来看，注入链已经表现出稳定性。

每次 `update_weights` 都重复出现相同模式：

- 一个 worker：
  - `tensor_keys=0`
  - 直接跳过 `add_lora`
- 另一个 worker：
  - `tensor_keys=372`
  - `normalized_keys=372`
  - `variant=normalized`
  - `parsed_layers=186`

这个模式在多个 step 上保持一致，没有再出现：

- `StopIteration`
- `parsed_layers=0`
- fallback 分支触发
- worker 崩溃

同时，训练已经持续推进到后续 step，并开始保存 checkpoint，例如：

- `global_step_5`

这说明：

- LoRA 动态注入已经不再阻塞 rollout
- `update_weights -> add_lora -> generate -> update_actor` 整条链路可以闭环运行

---

## 9. 为什么当前判断“注入链稳定”

当前判断稳定，主要基于四个信号。

### 9.1 解析结果稳定

每次都能看到：

```text
tensor adapter variant=normalized parsed_layers=186
```

`186` 对应 `372` 个 LoRA tensor 的 `A/B` 配对数量，这个数值稳定且合理。

### 9.2 分布式行为稳定

每次都是同一个模式：

- 一个 worker 收到空 tensor 并跳过
- 一个 worker 收到完整 tensor 并成功注入

没有表现出随机抖动或偶发失败。

### 9.3 训练主循环持续推进

训练可以连续推进多个 step，且：

- `timing_s/update_weights` 比较稳定
- 没有在 LoRA 更新阶段卡死或报错

### 9.4 checkpoint 已经开始产出

这说明当前修改不仅是“初始化时跑过”，而是已经进入真正的训练循环。

---

## 10. 当前仍需继续观察的点

虽然注入链已经稳定，但仍建议继续观察以下内容，它们更偏训练现象，而不是注入错误：

- 某些 step 上 `actor/loss`、`actor/grad_norm`、`critic/rewards/mean` 仍可能为 `0`
- `response_length/clip_ratio` 较高，很多 response 顶到 `2048`

这些现象更像：

- reward 稀疏
- rollout 输出过长
- 采样分布或截断配置需要继续调

它们和本次 LoRA 注入修复是两个层面的问题，不应混为一谈。

---

## 11. 涉及文件

本次修复和定位主要涉及以下文件：

- `verl/verl/utils/fsdp_utils.py`
- `verl/verl/workers/rollout/vllm_rollout/utils.py`
- `verl/verl/utils/vllm/utils.py`

如果后续需要继续收敛这条链路，建议优先从：

- rollout 侧空 tensor 分布
- vLLM `from_lora_tensors()` 的模块映射
- debug 日志精简

这三部分入手。

---

## 12. 最终结论

这次问题的本质不是：

- LoRA 路径错误
- LoRA 文件为空
- FSDP2 没有把 LoRA 收上来
- `default` adapter 名导致加载失败

真正的问题是：

- DAPO 的训练时 LoRA 动态注入链路，比 `LiveCodeBench/run.sh` 的本地 adapter 直加载更复杂
- 在 rollout/vLLM 侧，需要显式处理 tensor 注入的可观测性、空 tensor worker 和最小 key 规范化

修复完成后，当前链路已经表现出稳定特征：

- actor 能收集 LoRA
- rollout 能稳定接收并注入 LoRA
- vLLM 能稳定解析出 `186` 个 LoRA layer
- 训练主循环可以持续推进

因此，目前可以把这次问题视为：

- `DAPO + vLLM rollout` 动态 LoRA 注入链路已经基本修通

