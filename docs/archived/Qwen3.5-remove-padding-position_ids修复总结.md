# Qwen3.5 remove-padding 下 3D `position_ids` 修复总结

本文记录一次发生在 `Qwen3.5 + VERL + use_remove_padding=true` 场景下的 `position_ids` 形状错误排查过程，重点说明：

- 问题最初是怎么表现的
- 为什么 `ppo_micro_batch_size_per_gpu=1` 正常、`>1` 容易出错
- 中间几轮修复分别解决了什么、又遗漏了什么
- 最终稳定方案是什么

---

## 1. 问题现象

训练配置的关键点是：

- 模型：`Qwen3.5`
- `use_remove_padding=true`
- `train batch size=8`
- `ppo_micro_batch_size_per_gpu=2`

最初现象是：

- 当 `ppo_micro_batch_size_per_gpu=1` 时训练可以正常开始，且显存占用明显下降
- 当 `ppo_micro_batch_size_per_gpu>1` 时，在 actor 侧 `compute_log_prob` 或 `update_actor` 前向中报错
- 当 `use_remove_padding=false` 时不容易触发同类错误

典型错误包括：

```text
RuntimeError: The size of tensor a (3) must match the size of tensor b (8) at non-singleton dimension 0
```

以及：

```text
RuntimeError: The size of tensor a (5634) must match the size of tensor b (8) at non-singleton dimension 2
```

堆栈最终都落在 `transformers/models/qwen3_5/modeling_qwen3_5.py` 的 rotary embedding 路径。

---

## 2. 第一阶段判断：问题不在 Qwen3.5 本身，而在 `position_ids`

最早的核心判断是：

- Qwen3.5 需要多轴 `position_ids`
- VERL 在 `use_remove_padding=true` 时会把输入压平成 varlen 形式
- 一旦 `position_ids` 的轴顺序或 batch 组织方式出错，Qwen3.5 的 RoPE 会立刻报 shape mismatch

当时先补了调试日志，记录：

- remove-padding 前 `position_ids` 的逻辑形状
- flatten 后的形状
- Qwen3.5 最终收到的 `position_ids` 与 `inputs_embeds` 形状

这一阶段确认了一个关键事实：

- 报错不是由 `ppo_micro_batch_size_per_gpu` 本身直接引起的
- 真正的问题是 3D `position_ids` 在 micro-batch 切分、重建或 flatten 时被错误解释

---

## 3. 第二阶段修复：先修 flatten 路径

最初发现的第一个明确 bug 是：

- 对 3D jagged `position_ids` 直接取 `.values()` 不能得到逻辑上的 `(num_axes, total_tokens)`
- 它拿到的是 NestedTensor 的内部存储布局，而不是 Qwen3.5 期望的多轴位置编码布局

因此先做了第一轮修复：

- 在 `verl/verl/utils/tensordict_utils.py` 中新增 `flatten_3d_nested_position_ids(...)`
- 不再直接依赖 `.values()`，而是先转成 dense，再按每个 sample 的真实 `seq_len` 拼接
- FSDP、AutoModel、TorchTitan 三套 engine 的 remove-padding 路径都改成调用这个 helper

这一步解决了最初那类“batch=1 能跑、batch>1 立即炸掉”的 flatten 顺序问题。

但是这还不是全部。

---

## 4. 第三阶段修复：3D jagged `position_ids` 在切 micro-batch 时仍然会被重建错

后续日志显示，即使 flatten 逻辑已经修正，`position_ids` 在进入模型前仍可能变成错误形状，例如：

- 从正常的 `(2, 4, j...)`
- 变成异常的 `(2, 8, j...)`
- 或最终传进模型时变成 `(8, 1, 8)`

这说明问题还发生在更早阶段，也就是：

- `TensorDict` 被切成 micro-batch 时
- 3D `position_ids` 被重新打包成 jagged NestedTensor
- PyTorch 把错误的维度识别成了 ragged 维

于是做了第二轮修复：

- 修正 `chunk_tensordict(...)` 的 fallback 切片逻辑，只裁最后一个 ragged `seq` 维
- 在重建的 3D `position_ids` 上显式设置 `_ragged_idx = 2`
- 修复 `index_select_tensor_dict(...)`、`chunk_tensordict(...)` 里对 3D `position_ids` 的特殊处理

这一轮的核心目标是：

- 避免 3D jagged tensor 在“切 batch”这一步再次被错误解释

---

## 5. 第四阶段判断：仅靠 3D jagged NestedTensor 仍然不稳

再往后，日志继续暴露出一个更底层的问题：

- 即使使用 `nested_tensor_from_jagged(...)`
- 在当前 PyTorch / tensordict / 训练环境组合下
- 3D `position_ids` 仍然可能被构造成不符合预期的内部布局

一个典型信号是：

- 表面上 `position_ids.shape` 看起来像 `(2, 4, j...)`
- 但 `values_shape` 却已经不再对应逻辑上的多轴展开
- flatten 后可能只剩下极小的错误尺寸，例如 `(4, 8)`

这说明：

- 问题已经不只是“我们调用方式写错”
- 而是当前环境下 3D jagged NestedTensor 自身就不够稳

因此排查思路发生了收缩：

- 不再继续和 3D jagged NestedTensor 的内部行为博弈
- 改成在 RLHF 这条 remove-padding 路径里，直接绕开它

---

## 6. 最终稳定方案：3D `position_ids` 保持 dense，直到 flatten 时再按真实长度裁剪

最终稳定方案是：

- 对 RLHF 路径里的多轴 `position_ids`
- 不再保存成 3D jagged NestedTensor
- 而是直接保存成普通 dense tensor：

```text
(batch, num_axes, max_seq_len)
```

具体落点在：

- `verl/verl/workers/utils/padding.py`

这里会把每个 sample 的 `(num_axes, seq_len_i)`：

- 先右侧 pad 到同一个 `max_seq_len`
- 再 `torch.stack(...)` 成 dense tensor

对应地，helper 也同步扩展为同时支持：

- nested 3D `position_ids`
- dense 3D `position_ids`

具体包括：

- `verl/verl/utils/tensordict_utils.py`
  - `flatten_3d_nested_position_ids(...)`
  - `pad_3d_nested_position_ids(...)`

这样在 engine 的 remove-padding 路径里：

- 仍然可以用 `seq_lengths`
- 从 dense `(bs, axes, max_seq)` 中按真实长度切出每个 sample
- 再拼成 Qwen3.5 期望的 `(axes, total_tokens)`

这一步之后，训练和 validation 都恢复正常。

---

## 7. 修复过程中的几个关键认识

### 7.1 `ppo_micro_batch_size_per_gpu=1` 之所以“看起来正常”，很多时候只是把问题藏住了

当 micro-batch 只有一个 sample 时：

- 一些错误的 batch 组织方式不会立即暴露
- broadcasting 也可能在某些张量运算中临时掩盖维度问题

因此：

- `batch=1` 正常不能说明 3D `position_ids` 处理是正确的

### 7.2 `use_remove_padding=false` 稳定，不代表模型天然不需要修

`use_remove_padding=false` 路径之所以更稳，是因为：

- 它更接近普通 dense 输入
- 不依赖多次 jagged tensor 的拆分、重建、flatten

也就是说：

- 真正脆弱的是 3D `position_ids` 在 remove-padding 语义下的工程组织方式
- 而不是 Qwen3.5 模型本身不能训练

### 7.3 调试日志必须能覆盖到 Ray worker 内部

中间一度出现：

- 明明设置了 `VERL_DEBUG_POSITION_IDS=1`
- 但 worker 没有输出日志

原因是：

- driver 进程的环境变量并不会自动成为每个 Ray worker 的运行时判断依据

后来通过把 `debug_position_ids` 放进 batch metadata，才让调试信息稳定进入训练 worker。

---

## 8. 当前最终状态

到目前为止，针对这个问题的稳定修复思路可以总结为三条：

1. 3D `position_ids` 的 flatten 不能直接依赖 jagged `.values()`，必须按真实 `seq_len` 重建逻辑顺序
2. 3D `position_ids` 在 micro-batch 切分时不能再走通用的 3D jagged 重建路径
3. 在 RLHF remove-padding 路径中，最稳的做法是直接把多轴 `position_ids` 保持为 dense tensor，而不是 3D jagged NestedTensor

当前验证结果是：

- `use_remove_padding=true`
- `train batch size=8`
- `ppo_micro_batch_size_per_gpu=2`

训练已经能够正常推进，并通过 validation。

---

## 9. 这次实际涉及到的核心文件

- `verl/verl/workers/utils/padding.py`
- `verl/verl/utils/tensordict_utils.py`
- `verl/verl/workers/engine/fsdp/transformer_impl.py`
- `verl/verl/workers/engine/automodel/transformer_impl.py`
- `verl/verl/workers/engine/torchtitan/transformer_impl.py`
- `verl/verl/models/transformers/qwen3_5.py`
- `verl/verl/workers/engine_workers.py`

---

## 10. 后续建议

如果后面再接新的多模态模型，并且它也依赖多轴 `position_ids`，建议优先遵循下面的原则：

- 先确认模型实际期望的 `position_ids` 逻辑布局
- 不要默认把 3D 多轴位置编码直接塞进 jagged NestedTensor
- 只在确实验证过 API 语义稳定时，才依赖 3D jagged 的 `.values()` / `unbind()` / 重建行为
- 更保守的默认方案仍然是：保留 dense 多轴 tensor，在 remove-padding 入口再按真实长度 flatten
