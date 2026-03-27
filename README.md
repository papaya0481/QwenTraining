# 安装

运行以下命令 以克隆包含 `ms-swift` 子模块的仓库.
```
git clone --recurse-submodules <repo-url>
```

## Live Code Bench
需要安装 `datasets==3.6.0`[issue](https://github.com/LiveCodeBench/LiveCodeBench/issues/107)

如果遇到错
```
 Failed to import vLLM: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /data/wuli_error/miniconda3/envs/llmqw/lib/python3.11/lib-dynload/../.././libicui18n.so.78)
 ```
需要添加：
```
export LD_LIBRARY_PATH=/data/wuli_error/miniconda3/envs/llmqw/lib:$LD_LIBRARY_PATH
```