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