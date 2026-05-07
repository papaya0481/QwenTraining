# vLLM `#36976` 与 Qwen3.5 LoRA GDN 问题总结

本文总结 `vllm-project/vllm#36976` 解决的问题，以及它对当前仓库的参考价值。

这份总结主要回答三件事：

- `#36976` 修的到底是什么
- 它为什么会在 `Qwen3.5 + LoRA` 场景下触发
- 这和我们项目里遇到的问题有什么关系

---

## 1. 结论先行

`#36976` 的核心修法不是“补一个索引边界判断”，而是：

- 当 `Qwen3.5` 开启 `LoRA` 时
- 不再把 GDN 的 `in_proj_qkvz` 当成一个继续参与 LoRA packed mapping 的融合投影层
- 而是把它拆成独立的 `in_proj_qkv` 和 `in_proj_z`

这样做之后：

- LoRA 注入会对齐到真实可挂载的子模块
- 不再沿用 fused projection 的旧映射
- 从而避免 `Qwen3.5 + LoRA` 下的错误模块匹配或索引异常

从当前 vLLM 文档可以直接看到这一点：

- `Qwen3_5ForConditionalGeneration.packed_modules_mapping` 默认包含：
  - `in_proj_qkvz -> ["in_proj_qkv", "in_proj_z"]`
- 但 `update_packed_mapping(enable_lora=True)` 会显式：
  - `pop("in_proj_qkvz", None)`
  - 然后分别注册：
    - `in_proj_qkv -> ["in_proj_qkv"]`
    - `in_proj_z -> ["in_proj_z"]`

vLLM 源码注释写得很直接：

- 当启用 LoRA 时，GDN 使用分开的 `in_proj_qkv` 和 `in_proj_z`

---

## 2. 问题本质是什么

`Qwen3.5` 的 GDN 路径里，有一部分投影层在高性能实现中会使用 fused / packed 形式。

这在纯推理时通常没有问题，但到了 LoRA 场景会多出一个额外约束：

- LoRA 不是只关心“最终算子能不能跑”
- 它还要求框架正确识别“adapter 应该挂到哪个具体模块”

如果框架仍把某个融合层当成单个 packed target，而 LoRA 实际期待的是拆开的子模块，就容易出现两类问题：

1. 目标模块映射错误
2. 内部索引、偏移或模块查找和真实结构不一致

`#36976` 的思路，本质上就是承认这一点：

- fused 表达对推理友好
- 但不一定对 LoRA target mapping 友好

所以在 `enable_lora=True` 时，优先保证模块语义正确，而不是强行复用 fused mapping。

---

## 3. 它是怎么修的

从当前文档暴露出来的实现看，修复分两步：

### 3.1 默认仍保留 fused mapping

也就是：

- 非 LoRA 场景继续允许 `in_proj_qkvz`

这说明 vLLM 并没有否定 fused GDN 的实现本身，而是只在 LoRA 条件下改变映射策略。

### 3.2 一旦开启 LoRA，就切换到拆分后的 mapping

也就是：

- 删除 `in_proj_qkvz`
- 改为显式暴露：
  - `in_proj_qkv`
  - `in_proj_z`

这一步的工程意义非常明确：

- 让 LoRA 的模块发现逻辑和真实可注入结构保持一致
- 避免 packed module 在 LoRA 侧继续扮演“伪目标模块”

换句话说，修复点不在 LoRA tensor 本身，而在：

- `Qwen3.5/GDN` 模块结构
- `packed_modules_mapping`
- `LoRA target resolution`

这也是为什么这个 PR 能解决一类看起来像“LoRA 加载出错”或“LoRA 索引异常”的问题。

---

## 4. 这和我们项目的关系

当前仓库里，和这个主题最接近的历史问题有两类。

### 4.1 评测/侧载阶段的 `Qwen3.5 + vLLM + LoRA` 兼容问题

git 历史里可以看到一串明显相关的提交：

- `ef18f1a`：测试 `vLLM` 加载 `qwen3.5-lora`，并发现问题
- `053e617`：添加 `vLLM` 对 `Qwen3.5` 的支持
- `b05e3e7`：修复 `vLLM engine` 默认参数没有传入的问题
- `0c1bccd`：添加 LoRA 评测
- `adb7fbd`：让 `LiveCodeBench` 支持 LoRA 侧载

这说明我们当时遇到的并不只是“评测脚本没配好”，而是：

- `Qwen3.5`
- `vLLM`
- `LoRA`
- 评测/侧载路径

之间确实存在兼容性缺口。

`#36976` 虽然不一定和我们当时每一个报错完全相同，但它证明了一件非常重要的事：

- 上游 vLLM 自己也确认，`Qwen3.5` 的 fused GDN 投影在 LoRA 场景下需要特殊处理

因此，我们过去遇到的那类问题并不是偶然“环境没装好”，而是模型结构与 LoRA 注入语义之间确实存在工程冲突。

### 4.2 训练期的 LoRA 动态注入问题

当前仓库已经有：

- [DAPO-vLLM-LoRA动态注入修复总结](/home/ruixin/workspace/QwenTraining/docs/DAPO-vLLM-LoRA动态注入修复总结.md)

那份文档记录的是训练期 `TensorLoRARequest -> from_lora_tensors()` 这条动态注入链路。

它和 `#36976` 不是同一个 bug，但属于同一家族问题：

- 都发生在 `Qwen3.5 + vLLM + LoRA`
- 都说明“LoRA 在这类模型上不是通用线性层式的简单挂载”
- 都要求框架对真实模块结构有更准确的理解

可以把两者理解成：

- 我们的文档记录了“训练态动态注入链路”的坑
- `#36976` 记录了“模型结构映射/packed module 语义”的坑

两者叠在一起，基本能解释为什么这条链路比普通模型更脆弱。

---

## 5. 对当前仓库的建议

如果后续继续维护 `Qwen3.5 + vLLM + LoRA`，建议按下面思路处理。

### 5.1 优先确认 vLLM 版本是否已包含 `#36976`

如果包含：

- 优先直接升级到带该修复的版本
- 再验证我们现有的 LoRA 侧载、评测、动态注入路径

如果不包含：

- 可以考虑把这类“LoRA 下拆开 fused projection mapping”的思路做本地 backport

### 5.2 遇到 LoRA 加载异常时，不要只盯着 adapter 文件

对 `Qwen3.5` 来说，排查顺序最好是：

1. LoRA 文件本身是否完整
2. target modules 是否和模型真实结构一致
3. vLLM 是否仍在使用 fused / packed mapping
4. 当前路径是磁盘侧载，还是训练期 tensor 动态注入

原因很简单：

- 有些错误表面上像“LoRA 权重读错了”
- 实际上是“LoRA 被挂到了错误的模块语义上”

### 5.3 把“评测期问题”和“训练期问题”分开看

这类问题很容易被混在一起，但最好分两层排查：

- 评测/侧载阶段：
  - 重点看 vLLM 对 `Qwen3.5` 模型结构和 LoRA target 的支持
- 训练期动态注入阶段：
  - 重点看 tensor key、模块解析、worker 分布和 runtime 注入链路

这样更容易快速收敛根因。

---

## 6. 最终总结

`vLLM #36976` 给出的关键信号是：

- 在 `Qwen3.5` 这种带 fused GDN 投影的模型上
- LoRA 问题不一定是 adapter 本身坏了
- 更可能是“框架如何理解并映射真实模块结构”出了偏差

它的修法很有代表性：

- 不在错误的 packed abstraction 上继续修补
- 而是在 `enable_lora=True` 时退回到更明确、更稳定的子模块映射

这对当前仓库的价值在于：

- 它为我们过去遇到的 `Qwen3.5 + vLLM + LoRA` 兼容问题提供了上游佐证
- 也给出了后续升级或 backport 时最值得优先借鉴的修复方向

---

## 参考

- vLLM PR：<https://github.com/vllm-project/vllm/pull/36976>
- vLLM `Qwen3.5` 模型文档：<https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/>
- vLLM Releases：<https://github.com/vllm-project/vllm/releases>
