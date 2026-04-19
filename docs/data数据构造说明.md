# data 数据构造说明

本文只讨论当前仓库 `data/` 目录里的数据构造、清洗、验证与合并逻辑。

对应文件主要有：

- [data/SFT/clean_openthoughts.py](/home/ruixin/workspace/QwenTraining/data/SFT/clean_openthoughts.py)
- [data/SFT/merge_codegen1.py](/home/ruixin/workspace/QwenTraining/data/SFT/merge_codegen1.py)
- [data/RL/taco_v1.py](/home/ruixin/workspace/QwenTraining/data/RL/taco_v1.py)
- [data/code_excutor.py](/home/ruixin/workspace/QwenTraining/data/code_excutor.py)

---

## 1. `data/` 在项目里的职责

`data/` 不是训练器本身，而是训练前的数据工程层。

它解决三类问题：

1. 把不同来源的数据整理成统一 schema
2. 过滤掉结构坏样本和执行失败样本
3. 为 SFT 或 RL 准备可直接消费的数据集

这部分工作直接影响：

- 训练稳定性
- reward 可靠性
- 最终模型是否学到“推理 + 代码 + 可执行”这一整套行为

---

## 2. 当前 `data/` 目录的分工

### `clean_openthoughts.py`

负责把 `OpenThoughts-114k` 中 code domain 的样本清洗成适合当前项目的 SFT 风格数据。

重点是：

- 抽 code domain
- 限制样本长度
- 裁剪测试样例
- 用执行器验证 `deepseek_solution`
- 统一成 `system/user/assistant/test_cases` 结构

### `merge_codegen1.py`

负责把多个来源的数据集合并、去重、格式规范化，并做一些 schema 对齐。

重点是：

- 合并 OpenThoughts 与 codeforces 清洗数据
- 用 LSH 做近重复去重
- 统一代码块格式
- 统一字段 schema

### `taco_v1.py`

负责从 `TACO-verified` 构造更偏 RL/代码验证场景的数据。

重点是：

- 解析 `input_output`
- 选择可用 solution
- 用执行器验证 solution
- 选出可通过测试的实现
- 形成最终训练样本

### `code_excutor.py`

负责把“模型代码回答 + 测试样例”变成结构化执行结果。

它是整套 `data/` 清洗流程的基础设施。

---

## 3. 统一 schema 的基本思路

当前数据工程的核心不是简单“拼数据”，而是先把不同来源映射到统一结构。

常见统一字段包括：

- `system`
- `user`
- `assistant` 或 `answer`
- `test_cases`
- `fn_mode`
- `fn_name`
- `source_dataset`
- `task`

其中最关键的是：

- 题面统一成 prompt
- 答案统一成可抽取代码的 assistant/answer
- 测试样例统一成执行器可识别格式

如果这一步没做好，后面：

- 执行验证会乱
- reward 会漂
- 训练模板也会不稳定

---

## 4. `clean_openthoughts.py` 在做什么

### 4.1 数据来源与初筛

脚本从：

- `open-thoughts/OpenThoughts-114k`

里加载 `metadata/train`，然后：

- 只保留 `domain == "code"` 的样本
- 过滤掉 `deepseek_reasoning + deepseek_solution` 太长的样本

这一步的目的很直接：

- 只保留代码任务
- 控制上下文长度，避免异常长样本污染训练分布

### 4.2 测试样例裁剪

脚本会把 `test_cases` 最多裁到 10 个。

这样做的好处是：

- 验证更快
- 构造成本更低

代价是：

- 执行验证的覆盖面会变窄
- `pass` 标签不再等价于“通过原始全集测试”

所以它更适合作为训练数据清洗手段，而不是最终 benchmark 判定口径。

### 4.3 执行验证

脚本会动态加载：

- [data/code_excutor.py](/home/ruixin/workspace/QwenTraining/data/code_excutor.py)

然后对 `deepseek_solution` 做执行验证。

输出会写到：

- `_sample_passed`
- `_error`

这里本质上是在问：

- 这段 reasoning 最终给出的代码，能不能真的通过给定 testcase

### 4.4 最终格式化

通过验证后，样本会被整理成：

- `system`
- `user`
- `assistant`
- `test_cases`
- `source_dataset`
- `task`
- `pass`
- `errors`

同时脚本会：

- 把 reasoning 包装进 `<think> ... </think>`
- 从 `deepseek_solution` 中抽 fenced code block

这说明当前项目对 OpenThoughts 的使用方式不是原样保留，而是转成“推理 + 代码”的统一模板。

---

## 5. `merge_codegen1.py` 在做什么

### 5.1 合并多来源数据

这个脚本会把不同来源的数据集拼起来，当前可见的是：

- `BigfufuOuO/code_gen`
- `ZyWOuO/clean-codeforces`

合并时，会先做字段合法性与 schema 对齐。

### 5.2 `test_cases` 的脏数据拦截

脚本会过滤：

- `None`
- 空值
- 不能稳定 JSON 化的 `test_cases`

因为 testcase 一旦坏掉，执行器就无法给出可靠标签。

### 5.3 近重复去重

这个脚本里最明确的去重逻辑是：

- 对 `user` 字段做 LSH 去重

具体做法：

1. 文本转成字符 n-gram
2. 构造 `MinHash`
3. 用 `MinHashLSH` 查近邻
4. 保留没有命中近邻的样本

这个方法不是精确 dedup，但足够适合大规模数据的近重复过滤。

它的重点是：

- 去掉题面级近重复
- 降低相似题对训练分布的挤压

### 5.4 文本格式规范化

这个脚本还会做：

- 把 `</think>` 之后的代码块统一成 ```python
- 清理最外层引号问题

这属于格式噪声清理，目的不是提高语义质量，而是提高训练样本的一致性。

---

## 6. `taco_v1.py` 在做什么

### 6.1 数据来源

脚本从：

- `likaixin/TACO-verified`

加载数据。

这类数据更像“题目 + 测试 + 候选解”的结构化代码任务集。

### 6.2 结构清洗

脚本会先做一轮结构过滤：

- 没有 `input_output` 的样本丢弃
- `question` 含 HTTP 链接的样本丢弃
- 某些难度档位丢弃
- 无有效 solution 的样本丢弃

这一步是在保证：

- 样本能被解析
- 题面适合训练
- 不把明显脏样本送进执行验证

### 6.3 解析 `input_output`

`parse_input_output(...)` 会把题目的输入输出描述转成统一结构：

- `inputs`
- `outputs`
- `fn_name`

并进一步决定：

- `fn_mode = call_based`
- 或 `fn_mode = auto`

这一步直接影响执行器应该如何运行代码。

### 6.4 多候选解验证与选择

TACO 数据里可能有多个 solution。

当前脚本的策略不是全部保留，而是：

1. 逐个 solution 执行验证
2. 只要某个 solution 通过，就选它作为 `_selected_solution`
3. 若都不通过，则保留 fallback，但打上失败标记

这意味着它在做的是：

- “从多个候选参考解里挑一个真实可跑的”

而不是“所有参考解一视同仁全部保留”。

### 6.5 最终输出

格式化后，最终样本会包含：

- `system`
- `user`
- `answer`
- `test_cases`
- `fn_mode`
- `fn_name`
- `source_dataset`
- `task`
- `difficulty`
- `tags`
- `skill_types`

这更适合后续代码任务训练。

---

## 7. `code_excutor.py` 的地位

`data/` 目录里的所有高质量过滤，最终都离不开：

- [data/code_excutor.py](/home/ruixin/workspace/QwenTraining/data/code_excutor.py)

它的职责是：

1. 从模型回复里抽取代码
2. 统一 testcase 格式
3. 判断是 `call_based` 还是 `stdio`
4. 在受限子进程中运行代码
5. 返回逐 testcase 的结构化结果

这意味着当前数据构造最核心的“真过滤器”不是文本规则，而是执行验证。

很多文本上看起来合理的样本，只有真正跑过测试后，才知道能不能留下。

---

## 8. 当前数据构造的核心思想

如果把 `data/` 里的实现抽象一下，当前仓库的数据构造遵循的是这套顺序：

1. 先做 schema 清洗
2. 再做格式清洗
3. 再做近重复去重
4. 再做执行验证
5. 最后统一成训练可消费模板

这样做的原因是：

- 结构错误样本先删掉，节约执行成本
- 近重复先处理，避免在重复样本上浪费验证资源
- 只有进入执行验证的样本，才有机会成为高质量监督数据

---

## 9. 当前实现的优势

### 1. 执行验证贯穿数据工程

不是只靠规则文本过滤，而是用测试样例真正验证代码质量。

### 2. 兼容多种题型

当前执行器能同时处理：

- `stdio`
- `call_based`

这让不同平台题目更容易混合进同一训练集。

### 3. 合并、去重、清洗链条比较完整

从 OpenThoughts 到 codeforces，再到 TACO，当前已经具备一套基本可复用的数据工程管道。

---

## 10. 当前实现的限制

### 1. CoT 质量没有单独模型打分

当前更像“结果导向过滤”：

- 有代码
- 能执行
- 能过测试

而不是对 reasoning 文本做细粒度质量评分。

### 2. testcase 裁剪会改变监督口径

裁到 10 个 testcase 虽然实用，但也会降低验证覆盖率。

### 3. 去重目前主要是题面级

`merge_codegen1.py` 主要对 `user` 做近重复过滤，尚不是 testcase 级、AST 级或语义 embedding 级的多视角 dedup。

### 4. 缓存路径偏硬编码

部分脚本把缓存和保存路径直接写在脚本里，迁移环境时需要手工改。

---

## 11. 如何理解这部分代码

对当前项目来说，`data/` 最好被理解为：

- 训练前的“样本可信化”工程

而不是单纯的数据下载脚本。

它做的不是“多收数据”，而是“把能用于代码训练的数据变得更可靠、更统一、更可执行”。

这部分越稳：

- 后面的 reward 越稳
- 训练过程越不容易学到脏模式
- 最终模型越可能真正学会“推理到代码”的可执行行为

---

## 12. 文件索引

- OpenThoughts 清洗
  [data/SFT/clean_openthoughts.py](/home/ruixin/workspace/QwenTraining/data/SFT/clean_openthoughts.py)
- 合并与近重复去重
  [data/SFT/merge_codegen1.py](/home/ruixin/workspace/QwenTraining/data/SFT/merge_codegen1.py)
- TACO 数据构造
  [data/RL/taco_v1.py](/home/ruixin/workspace/QwenTraining/data/RL/taco_v1.py)
- 执行器
  [data/code_excutor.py](/home/ruixin/workspace/QwenTraining/data/code_excutor.py)
