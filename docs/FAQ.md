# FAQ

## Q: overlong buffer 是如何实现的？

### 短答案

`overlong buffer` 的核心思想不是“只要回答长就罚”，而是“在接近最大生成长度时，预留一段缓冲区；只有进入这段缓冲区以后，才逐步施加长度惩罚”。  
这样做的目标是把“允许模型正常展开推理”和“防止模型拖到长度上限”这两件事分开处理。

### 先说原理

在代码任务里，回答过短和回答过长都可能有问题：

- 太短时，模型可能还没写完关键逻辑、解释或者代码主体，正确率会下降。
- 太长时，模型可能开始重复、兜圈子、无效展开，甚至把生成长度顶到 `max_response_length`，浪费 rollout 预算，也让训练更不稳定。

如果我们从第 1 个 token 开始就对长度施加惩罚，模型会过早学会“少说点”，这通常会伤害需要中等或较长推理链的任务。  
但如果完全不惩罚，模型又容易学到“多写总没坏处”，最后把长度推到上限附近。

所以 `overlong buffer` 的设计是：

- 在最大响应长度 `max_resp_len` 之前，先留一段“安全区”。
- 只要响应长度还在安全区内，就不因为“写得长一些”而额外扣分。
- 一旦进入靠近上限的那段 buffer 区域，就按超出的比例线性扣分。

它本质上是在 reward 里加入一个“接近长度上限的风险成本”。

### 数学上怎么理解

设：

- `max_resp_len` 是允许的最大 response 长度
- `buffer_len` 是 `overlong_buffer_cfg.len`
- `penalty_factor` 是 `overlong_buffer_cfg.penalty_factor`
- `L` 是当前样本的实际 response 长度

先定义一个“开始惩罚的阈值”：

$$
L_{start} = max\_resp\_len - buffer\_len
$$

然后计算超出安全区的长度：

$$
exceed = L - L_{start}
$$

最终的 overlong 惩罚是：

$$
overlong\_reward = \min\left(- \frac{exceed}{buffer\_len} \times penalty\_factor,\ 0\right)
$$

因此有三个很直观的阶段：

1. 当 $L \le L_{start}$ 时，没有惩罚。  
也就是模型还在安全区内，可以正常展开回答。

2. 当 $L_{start} < L < max\_resp\_len$ 时，惩罚线性增大。  
也就是越接近长度上限，扣分越多。

3. 当 $L = max\_resp\_len$ 时，惩罚达到大约 $-penalty\_factor$。  
这代表“把回答写满到上限”会有一个明确成本。

### 为什么要用 buffer，而不是直接按总长度惩罚

这里最重要的不是实现细节，而是训练信号的形状。

如果直接按总长度惩罚，模型会把“写长”本身当成坏事。  
但在代码生成和推理任务里，真正坏的往往不是“稍微长一点”，而是“长到快撞上长度上限，还在继续拖”。

所以 `overlong buffer` 更像一种“尾部风险控制”：

- 前半段长度基本不管，让模型保留完成任务所需的表达空间。
- 只在末端区域提高惩罚斜率，防止模型把 rollout 预算耗在无效尾巴上。

这和很多系统里的 soft limit 很像：

- 不是一超过某个正常范围就立刻强制失败；
- 而是在危险区间里逐渐提高代价，逼着模型学会更有效率地结束。

### 在这个仓库里是怎么落地的

当前实现是在 reward manager 里完成的，而不是在自定义 reward 函数 `compute_score` 里完成。

以 DAPO reward manager 为例，流程是：

1. 先计算任务本身的 reward。  
例如代码任务里，先根据 test cases 得到 `score`、`acc`、`traj_reward` 等。

2. 再读取当前样本的有效 response 长度 `valid_response_length`。

3. 如果启用了 `overlong_buffer_cfg.enable`，就根据上面的公式额外计算一个负的 `overlong_reward`。

4. 把这个惩罚加到最终 reward 上：

$$
reward = score + overlong\_reward
$$

对应实现可以看：

- [verl/verl/workers/reward_manager/dapo.py](/home/ruixin/workspace/QwenTraining/verl/verl/workers/reward_manager/dapo.py:121)
- [scripts/plugins/verpo_reward_manager.py](/home/ruixin/workspace/QwenTraining/scripts/plugins/verpo_reward_manager.py:221)

从这个顺序也能看出来，`overlong buffer` 不是用来替代任务 reward 的，而是一个附加正则项。  
任务做对仍然重要，但如果模型总是把回答拉到极限长度，它会为这种低效率付出代价。

### 它和 `filter_overlong_prompts` 不是一回事

这两个名字很像，但作用层次不同：

- `filter_overlong_prompts` 是数据侧过滤。它主要处理“prompt 本身太长，连正常生成空间都不够”的样本。
- `overlong buffer` 是 reward 侧惩罚。它处理的是“response 虽然还能生成，但已经长到接近上限”的情况。

也就是说：

- 前者是在样本进入训练前做输入控制；
- 后者是在样本完成 rollout 后做输出约束。

这两者一起使用时，效果通常更稳：

- 先避免一开始就几乎塞满上下文窗口的坏样本；
- 再防止模型在可用生成空间里无限拖长回答。

### 怎么理解配置里的几个参数

以你的配置为例：

- `max_resp_len = 10240`
- `overlong_buffer_cfg.len = 2048`
- `penalty_factor = 1.0`

则开始惩罚的长度阈值是：

$$
10240 - 2048 = 8192
$$

这表示：

- 回答长度不超过 8192 时，没有 overlong 惩罚。
- 从 8192 到 10240 之间，惩罚线性增加。
- 到 10240 时，惩罚大约是 `-1.0`。

从行为上看，这相当于说：

- “给模型 8192 token 的自由发挥空间。”
- “最后 2048 token 是危险缓冲区，越往里走越要付成本。”

### 这个设计想解决什么训练问题

它主要在解决两个问题：

1. 防止长度投机  
有些策略会通过拉长回答来“赌”更多中间步骤，但这些步骤不一定真正提高正确率。

2. 提高 rollout 效率  
当很多样本都靠近长度上限时，吞吐、显存和 step 时间都会被拖慢。`overlong buffer` 相当于把这部分系统成本显式地反映到 reward 里。

所以从训练观点看，它不是单纯的“格式规则”，而是一种把“长度预算是稀缺资源”这件事注入优化目标的方式。

### 什么时候应该调大或调小

可以用下面这个直觉：

- 如果模型经常在还没完成任务前就被迫收得太短，可以适当增大 `buffer_len` 或减小 `penalty_factor`。
- 如果模型经常生成一大段冗余尾巴，并且频繁顶到 `max_response_length`，可以适当减小 `buffer_len` 或增大 `penalty_factor`。

一般来说：

- `buffer_len` 决定“从哪里开始觉得危险”
- `penalty_factor` 决定“危险以后罚得有多重”

前者更像位置参数，后者更像强度参数。
