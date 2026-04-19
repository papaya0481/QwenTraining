# 3D `position_ids` 在 micro-batch 切分后的 bug 分析与修复

本文记录一次和 `Qwen-VL / Qwen3.5` 多维 `position_ids` 相关的训练 bug，包含：

- 现象与触发条件
- 为什么“没有输入视频”也会遇到
- 和 issue `#5554` 的关系
- 代码层面的根因
- 本次修复内容
- 如何在本地运行 CPU 回归测试

相关 issue：

- https://github.com/verl-project/verl/issues/5554

---

## 1. 现象

在以下两类场景中，训练会报类似错误：

```text
RuntimeError: The size of tensor a (3) must match the size of tensor b (8) at non-singleton dimension 0
```

我们实际观察到的触发条件是：

- `use_dynamic_bsz=True`，且实际切出来的训练 micro-batch 含有 2 条及以上样本
- `use_dynamic_bsz=False`，但 `ppo_micro_batch_size_per_gpu >= 2`

而当：

- `use_dynamic_bsz=False`
- `ppo_micro_batch_size_per_gpu=1`

时，通常不会触发。

这说明问题并不是单纯由 `dynamic batch size` 开关导致，而是由“训练阶段出现 `bs>=2` 的 micro-batch”触发。

---

## 2. 为什么没有输入视频也会出现

这个问题容易让人误以为必须“真的输入视频”才会触发，但实际并不是这样。

对于 `Qwen-VL` 系列和 `Qwen3.5` 这一类模型，数据侧可能会构造 3D 的 `position_ids`，用于多模态或 mRoPE 位置编码路径。项目里已有相关逻辑：

- 多模态样本会构造 `(4, seq_len)` 形状的单样本 `position_ids`
- collate 后会变成 3D jagged nested tensor
- 后续训练路径需要依赖 nested tensor 的 ragged 轴元信息来正确解释这个张量

也就是说，是否“输入了视频文件”并不是唯一条件。只要某条数据经过处理后走到了 3D `position_ids` 路径，后续 micro-batch 切分就可能暴露这个 bug。

---

## 3. 和 issue #5554 的关系

`#5554` 的表象是：

- batch 中混合了不同视频时长
- 训练过程中出现 `position_ids` 相关 shape mismatch

那个 issue 的直接触发器是“不同视频长度混合”，但和我们这里的底层原因是一致的：

- 3D jagged 的 `position_ids`
- 在重新组 batch 或切 micro-batch 时被重建
- 重建后丢失了 ragged 轴信息
- 后续模型前向把某一维误当成 batch 维，最终在 RoPE 里报错

所以可以把它理解成同一个底层 bug 的两种触发方式：

- issue `#5554`：由多模态样本混合触发
- 我们当前场景：由训练 micro-batch 中 `bs>=2` 触发

---

## 4. 根因分析

### 4.1 关键点不是 `dynamic bsz`，而是“切分后重建 TensorDict”

训练前，代码会对 3D `position_ids` 做一次修正：

- `verl/verl/utils/tensordict_utils.py`
- `maybe_fix_3d_position_ids(...)`

核心是手动补：

```python
data["position_ids"]._ragged_idx = 2
```

这个字段非常关键。它告诉 PyTorch 这个 3D jagged nested tensor 的 ragged 轴在什么位置。

### 4.2 问题出在 micro-batch 切分后的重建

后续训练会把原始 batch 切成多个 micro-batch：

- 动态 batch 路径使用 `index_select_tensor_dict(...)`
- 静态 micro-batch 路径使用 `chunk_tensordict(...)`

这两个函数内部都会重新调用：

```python
torch.nested.as_nested_tensor(...)
```

来重建新的 nested tensor。

问题在于，PyTorch 重建出的新 nested tensor 不会自动保留 `_ragged_idx` 这类自定义元信息。

结果就是：

1. 原始 batch 里的 `position_ids` 是正确的
2. 一旦被切成新的 micro-batch
3. `position_ids` 虽然看起来还是 nested tensor
4. 但 `_ragged_idx=2` 已经丢了
5. 后续 FSDP engine 再去读这个 3D nested tensor 时，就可能把维度解释错

### 4.3 为什么 `bs=1` 往往不报错

因为 `bs=1` 时，很多维度错误不会立刻显性暴露。

一旦 `bs>=2`：

- batch 维不再是退化维
- 错误的 ragged 轴解释会直接把“mRoPE 维度”与“batch 维”混淆
- 最终在 Qwen3.5 的 rotary embedding 路径中出现 `3 vs 8` 这种错误

因此现象上就表现为：

- `ppo_micro_batch_size_per_gpu=1` 正常
- `ppo_micro_batch_size_per_gpu=2` 出错
- `use_dynamic_bsz=True` 时，只要切出来的某个 micro-batch `bs>=2`，也会出错

---

## 5. 本次修复

修复原则很简单：

- 既然问题出在 nested tensor 重建时丢失元信息
- 那就在所有重建位置把必要元信息补回去

### 5.1 修复文件

主要修改：

- [verl/verl/utils/tensordict_utils.py](/Users/thebug/Desktop/new/verls/QwenTraining/verl/verl/utils/tensordict_utils.py:162)

新增辅助函数：

- `_copy_nested_tensor_metadata(src, dst)`

作用：

- 如果源 nested tensor 带有 `_ragged_idx`
- 则在重建出的目标 nested tensor 上恢复该字段

### 5.2 修复覆盖的路径

本次修复覆盖了以下几个重建 nested tensor 的关键路径：

1. `concat_nested_tensors(...)`
2. `chunk_tensordict(...)`
3. `index_select_tensor_dict(...)`

这样可以同时覆盖：

- `ppo_micro_batch_size_per_gpu > 1` 的静态切 batch
- `use_dynamic_bsz=True` 的动态切 batch
- 某些后续拼接场景

---

## 6. 补充的回归测试

新增或增强的测试主要有两类。

### 6.1 `TensorDict` 工具层测试

文件：

- [verl/tests/test_protocol_v2_on_cpu.py](/Users/thebug/Desktop/new/verls/QwenTraining/verl/tests/test_protocol_v2_on_cpu.py:781)

覆盖内容：

- `chunk_tensordict(...)` 后 `position_ids._ragged_idx` 仍然存在
- `index_select_tensor_dict(...)` 后 `position_ids._ragged_idx` 仍然存在

### 6.2 `prepare_micro_batches(...)` 路径测试

文件：

- [verl/tests/utils/test_prepare_micro_batches_with_group_size.py](/Users/thebug/Desktop/new/verls/QwenTraining/verl/tests/utils/test_prepare_micro_batches_with_group_size.py:34)

覆盖内容：

- 动态 batch 路径下，3D `position_ids` 元信息不丢失
- 静态 `micro_batch_size_per_gpu=2` 路径下，3D `position_ids` 元信息不丢失

---

## 7. 如何在本地运行 CPU 测试

下面这些命令都建议在仓库根目录执行：

```bash
cd /Users/thebug/Desktop/new/verls/QwenTraining
```

### 7.1 先做最小语法检查

```bash
python3 -m py_compile \
  verl/verl/utils/tensordict_utils.py \
  verl/tests/test_protocol_v2_on_cpu.py \
  verl/tests/utils/test_prepare_micro_batches_with_group_size.py
```

如果这一步报错，优先把报错贴给我。

### 7.2 跑 `TensorDict` 工具层回归

```bash
python3 -m pytest verl/tests/test_protocol_v2_on_cpu.py -k "chunk_tensordict or index_select_tensordict_preserves_3d_position_ids_metadata" -q
```

### 7.3 跑 micro-batch 路径回归

```bash
python3 -m pytest verl/tests/utils/test_prepare_micro_batches_with_group_size.py -k "preserves_3d_position_ids" -q
```

### 7.4 一次性跑这两组相关测试

```bash
python3 -m pytest \
  verl/tests/test_protocol_v2_on_cpu.py \
  verl/tests/utils/test_prepare_micro_batches_with_group_size.py \
  -k "position_ids or preserves_3d_position_ids or chunk_tensordict" \
  -q
```

### 7.5 如果你的环境不是当前 shell 的 `python3`

比如你训练时用的是某个 conda 环境，建议先切到同一个环境，再执行：

```bash
which python3
python3 -c "import torch; import pytest; print(torch.__version__)"
```

如果这里导入失败，说明不是代码问题，而是测试环境没激活对。

---

## 8. 给我的反馈建议

你跑完后，优先把下面几类信息发给我：

1. 测试命令本身
2. `pytest` 最后的通过/失败摘要
3. 如果失败，把完整 traceback 发我
4. 如果测试通过，再补一条真实训练配置下的结果：
   `use_dynamic_bsz=True` 且 `ppo_max_token_len_per_gpu=8192` 是否还报原来的错

如果你愿意进一步验证训练侧，建议优先试这两组：

1. `use_dynamic_bsz=False`, `ppo_micro_batch_size_per_gpu=2`
2. `use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=8192`

因为这两组正好对应这次修复最核心的两条路径。

---

## 9. 当前结论

这次 bug 的本质不是 OOM，也不是单纯的 `dynamic batch size` 问题，而是：

- 3D nested `position_ids`
- 在 micro-batch 切分/重建后
- 丢失了 `_ragged_idx=2`
- 导致后续前向把维度解释错

本次修复通过在 nested tensor 重建时显式保留元信息，统一修复了：

- 动态 batch
- 静态 micro-batch
- 以及相关拼接路径

如果 CPU 回归测试通过，下一步就建议直接回到真实训练配置上做复现验证。
