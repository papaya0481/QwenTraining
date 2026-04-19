# LiveCodeBench 评测链路说明

本文只讨论本项目中的 `LiveCodeBench/` 评测链路，不讨论训练期 reward，也不讨论 `data/` 数据清洗。

---

## 1. 它在项目里的角色

在这个仓库里，`LiveCodeBench` 是独立于训练流程的一套 benchmark harness，用来做代码能力评测。

它和训练期的关系是：

- 训练期：`scripts/dapo/*` + `verl` + 自定义 reward，目标是优化模型。
- 评测期：`LiveCodeBench/`，目标是用统一 benchmark 口径统计 `pass@1`、`pass@5` 等指标。

所以这里的“评测链路部署”，应该理解成：

- 如何把模型输出接入 `LiveCodeBench`
- 如何生成评测输入
- 如何提取代码
- 如何跑测试
- 如何汇总分数

而不是训练时 reward 的执行链路。

---

## 2. 目录结构与关键入口

和评测链路最相关的文件有：

- 总入口
  [LiveCodeBench/lcb_runner/runner/main.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/runner/main.py)
- 自定义结果评测入口
  [LiveCodeBench/lcb_runner/runner/custom_evaluator.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/runner/custom_evaluator.py)
- 参数定义
  [LiveCodeBench/lcb_runner/runner/parser.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/runner/parser.py)
- 场景路由
  [LiveCodeBench/lcb_runner/runner/scenario_router.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/runner/scenario_router.py)
- Code Generation benchmark 定义
  [LiveCodeBench/lcb_runner/benchmarks/code_generation.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/benchmarks/code_generation.py)
- Code Generation 指标计算
  [LiveCodeBench/lcb_runner/evaluation/compute_code_generation_metrics.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/evaluation/compute_code_generation_metrics.py)
- 时间窗口汇总
  [LiveCodeBench/lcb_runner/evaluation/compute_scores.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/evaluation/compute_scores.py)
- 提示词与答案抽取
  `LiveCodeBench/lcb_runner/prompts/*`
  `LiveCodeBench/lcb_runner/utils/extraction_utils.py`

如果当前项目主要关心代码生成评测，那么最重要的路径其实是：

1. `runner/main.py`
2. `runner/scenario_router.py`
3. `benchmarks/code_generation.py`
4. `evaluation/compute_code_generation_metrics.py`

---

## 3. 高层流程

以 `codegeneration` 场景为例，整条评测链路可以拆成六步。

### 第一步：确定评测场景与数据版本

`main.py` 会先读参数，再通过 `scenario_router.build_prompt_benchmark(args)` 决定：

- 当前是什么场景
- 要加载哪份 benchmark
- 用哪套 prompt 格式

对代码生成任务，默认场景是：

- `Scenario.codegeneration`

同时还会读取：

- `release_version`
- `start_date`
- `end_date`
- `not_fast`

这些参数会决定最终评测集范围。

---

### 第二步：加载 benchmark 数据

代码生成 benchmark 由：

- [code_generation.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/benchmarks/code_generation.py)

负责加载。

这里有两个主要入口：

- `load_code_generation_dataset()`
- `load_code_generation_dataset_not_fast()`

默认更常用的是：

- `livecodebench/code_generation_lite`

如果加上 `--not_fast`，则会切到：

- `livecodebench/code_generation`

`lite` 版本是剪裁过的测试集，跑得更快，适合日常评测；完整版则更重。

每个题目会被包装成 `CodeGenerationProblem`，其中包含：

- 题面
- 平台
- `question_id`
- `contest_id`
- `contest_date`
- `starter_code`
- `difficulty`
- public/private test cases
- metadata

这里一个很重要的设计是：每道题都带发布时间 `contest_date`，所以可以按时间窗口做抗污染评测。

---

### 第三步：构造 prompt 并调用模型

在：

- [scenario_router.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/runner/scenario_router.py)

中，代码生成场景会选用：

- `format_prompt_generation`

然后 `main.py` 再通过 `build_runner(args, model)` 选择实际 runner。

也就是说，模型调用层和 benchmark 本身是解耦的。

支持两种主要接入方式：

#### 方式一：直接由 LiveCodeBench 调模型

通过：

- `python -m lcb_runner.runner.main ...`

让框架自己完成：

- prompt 构造
- 调模型
- 保存原始输出
- 抽取代码
- 评测

这适合：

- 直接评估 API 模型
- 直接评估 vLLM 本地模型

#### 方式二：外部先生成，再交给 custom evaluator

通过：

- `python -m lcb_runner.runner.custom_evaluator --custom_output_file ...`

把已经生成好的结果交给 LCB 评测。

这更适合当前项目，因为训练好的模型往往已经有独立推理脚本，不一定要绑定在 LCB 自带 runner 上。

---

## 4. custom evaluator 的实际意义

对你这个仓库来说，`custom_evaluator.py` 往往比 `main.py` 更关键。

原因很直接：

- 训练后的模型，可能不想直接按 LCB 内置 prompt 再调用一次 runner
- 你可能已经有自己的推理产物
- 你只需要“按 LCB 口径打分”

此时最稳定的做法是：

1. 先用你自己的推理脚本生成答案
2. 整理成 LCB 要求的 JSON 格式
3. 再交给 `custom_evaluator.py`

对 `codegeneration` 场景，它要求每个样本至少提供：

- `question_id`
- `code_list`

也就是类似：

```json
[
  {"question_id": "id1", "code_list": ["code1", "code2"]},
  {"question_id": "id2", "code_list": ["code1", "code2"]}
]
```

这里的 `code_list` 通常表示同一道题的多次采样结果。

这一步之后，LCB 会：

- 校验样本数是否和 benchmark 一致
- 按 `question_id` 排序
- 调统一评测函数计算指标

这能保证你的外部推理流程和 benchmark 评测口径分离。

---

## 5. 输出抽取与标准化

模型的原始输出不一定直接就是纯代码，因此 LCB 会先做提取。

对应逻辑在：

- [scenario_router.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/runner/scenario_router.py)

对 `codegeneration` 场景，会调用：

- `extract_code(output, model.model_style)`

也就是说，评测时真正送去跑测试的不是完整原始回复，而是“抽取后的代码”。

这件事非常重要，因为：

- 模型可能输出解释文字
- 可能有 markdown 包裹
- 可能有模型家族特定格式

所以 LiveCodeBench 的评测口径并不是“原样执行字符串”，而是“先按模型风格抽取，再评测”。

如果你用自定义推理脚本，最好确认：

- 你输出的格式能被 LCB 的抽取器稳定识别
- 或者你直接在 `code_list` 里放纯代码，减少抽取误差

---

## 6. 代码生成场景下，测试是怎么跑的

代码生成指标计算在：

- [compute_code_generation_metrics.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/evaluation/compute_code_generation_metrics.py)

高层逻辑是：

1. 把 benchmark 中每道题转成 evaluation sample
2. 对每个候选代码运行 correctness check
3. 得到每道题每个样本的测试结果
4. 再由 `pass_k_utils` 汇总成 `pass@1`、`pass@5` 等指标

在 `CodeGenerationProblem.get_evaluation_sample()` 里，LCB 会把一道题转成：

- `input_output.inputs`
- `input_output.outputs`
- `fn_name`

这说明它本质上也是把题目转成统一的 I/O 测试格式，再交给评测器。

对每个候选解，`check_correctness()` 会：

- 启一个独立进程
- 设置一个全局超时
- 调 `run_test(...)`
- 收集所有 testcase 的通过情况

所以 LCB 的评测口径是：

- 每个候选程序真正执行测试
- 不是只做字符串匹配
- 不是只看 public testcase
- 会综合题目提供的测试集合做判定

---

## 7. 结果文件的三层结构

运行 `main.py` 或 `custom_evaluator.py` 后，通常会得到三类输出。

### 第一类：原始输出文件

例如：

- `...output.json`

这里面一般保存：

- 每个题的原始 `output_list`
- 抽取后的 `code_list`

它更像“模型生成记录”。

### 第二类：汇总评测结果

例如：

- `..._eval.json`

这里面一般保存：

- 总体 `pass@1`
- `pass@5`
- 以及 detail 统计

它更像“实验汇总指标”。

### 第三类：逐题评测明细

例如：

- `..._eval_all.json`

这里面会保存每道题：

- 生成内容
- 抽取代码
- `graded_list`
- 每题 `pass@1`
- metadata

这份文件最适合做误差分析，因为它保留了题目级别粒度。

---

## 8. 时间窗口评测

LCB 的一个核心价值是按发布时间做过滤。

除了在加载 benchmark 时用：

- `--start_date`
- `--end_date`

还可以在已有 `eval_all_file` 基础上，通过：

- [compute_scores.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/evaluation/compute_scores.py)

做时间窗口汇总。

仓库里的：

- [LiveCodeBench/compute.sh](/home/ruixin/workspace/QwenTraining/LiveCodeBench/compute.sh)

本质上就是这个用途的一个简化调用。

这个机制的重要性在于：

- 你可以只看某段时间之后的新题
- 可以降低训练数据或预训练语料污染 benchmark 的风险
- 更适合比较不同模型在“新题”上的泛化能力

---

## 9. 对当前项目最实用的接法

如果结合这个仓库的训练流程，最实用的 LiveCodeBench 评测方案通常是：

1. 训练完成后，用你自己的推理脚本在 LCB benchmark 上生成代码
2. 生成一个符合 `custom_evaluator.py` 要求的 JSON 文件
3. 用 `custom_evaluator.py` 做统一评测
4. 再用 `compute_scores.py` 按时间窗口切分结果

原因是：

- 训练模型的调用方式、模板、batching 逻辑，通常和 LCB 内置 runner 不完全一致
- benchmark 评测最好只负责“打分”，不要同时承担“推理逻辑试验场”的角色

这样职责更清楚：

- 你的推理脚本负责生成
- LiveCodeBench 负责按官方口径评测

---

## 10. 当前需要注意的点

### 1. `LiveCodeBench` 不等于训练 reward

LCB 是 benchmark，不是训练期 reward 模块。它们都依赖执行测试，但用途完全不同。

### 2. `lite` 和完整版口径不同

默认 `code_generation_lite` 更快，但和完整版不是完全同一测试规模。写报告时要标清是否用了 `--not_fast`。

### 3. 抽取器会影响最终成绩

如果模型输出格式和抽取器不兼容，可能不是代码错，而是代码没被正确抽出来。

### 4. 多样本评测要区分 `n`

`pass@1`、`pass@5` 依赖每题生成几个候选程序。比较模型时，必须统一采样数和温度。

---

## 11. 文件索引

- 官方说明
  [LiveCodeBench/README.md](/home/ruixin/workspace/QwenTraining/LiveCodeBench/README.md)
- 主运行入口
  [LiveCodeBench/lcb_runner/runner/main.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/runner/main.py)
- 自定义评测入口
  [LiveCodeBench/lcb_runner/runner/custom_evaluator.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/runner/custom_evaluator.py)
- 场景路由
  [LiveCodeBench/lcb_runner/runner/scenario_router.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/runner/scenario_router.py)
- 代码生成 benchmark
  [LiveCodeBench/lcb_runner/benchmarks/code_generation.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/benchmarks/code_generation.py)
- 代码生成评测
  [LiveCodeBench/lcb_runner/evaluation/compute_code_generation_metrics.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/evaluation/compute_code_generation_metrics.py)
- 时间窗口评分
  [LiveCodeBench/lcb_runner/evaluation/compute_scores.py](/home/ruixin/workspace/QwenTraining/LiveCodeBench/lcb_runner/evaluation/compute_scores.py)
