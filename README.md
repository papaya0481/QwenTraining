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


## 其他
1. 关于多卡运行的问题[issue](https://github.com/modelscope/ms-swift/issues/3991)
2. Qwen3.5的最佳时实践[Qwen3.5](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html)
