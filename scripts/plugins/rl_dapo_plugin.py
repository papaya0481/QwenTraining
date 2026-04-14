# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# 动态导入以兼容 VS Code 分析环境与实际训练环境不一致的情况。
swift_dataset = importlib.import_module('swift.dataset')
swift_rewards = importlib.import_module('swift.rewards')

DatasetMeta = swift_dataset.DatasetMeta
MessagesPreprocessor = swift_dataset.MessagesPreprocessor
SubsetDataset = swift_dataset.SubsetDataset
load_dataset = swift_dataset.load_dataset
register_dataset = swift_dataset.register_dataset

ORM = swift_rewards.ORM
orms = swift_rewards.orms


class CodeGen1RLPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        system = row.get('system', '')
        user = row.get('user', '')

        # RLHF仅使用prompt侧信息，assistant会在rollout阶段由模型采样生成。
        row['messages'] = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]

        # 保留奖励函数所需字段。
        row['test_cases'] = row.get('test_cases')
        row['fn_mode'] = row.get('fn_mode', 'auto')
        row['task'] = row.get('task', 'code_generation')
        row['source_dataset'] = row.get('source_dataset', '')
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        hf_dataset_id='BigfufuOuO/codegen1_merged_clean',
        dataset_name='codegen1_train_rl',
        preprocess_func=CodeGen1RLPreprocessor(),
        subsets=[
            SubsetDataset('rl', subset='rl', split=['train']),
        ],
    )
)


def _ensure_list(value: Any, n: int) -> List[Any]:
    if isinstance(value, list):
        if len(value) == n:
            return value
        if len(value) == 1 and n > 1:
            return value * n
        if len(value) > n:
            return value[:n]
        if len(value) < n:
            return value + [value[-1] if value else None] * (n - len(value))
    return [value] * n


def _parse_test_cases(raw_test_cases: Any) -> Any:
    if raw_test_cases is None:
        return None
    if isinstance(raw_test_cases, str):
        text = raw_test_cases.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None
    return raw_test_cases


def _load_executor_cls():
    executor_path = Path(__file__).resolve().parents[1] / 'data' / 'code_excutor.py'
    spec = importlib.util.spec_from_file_location('code_excutor', executor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load code executor from: {executor_path}')
    code_executor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code_executor_module)
    return code_executor_module.ModelResponseCodeExecutor


_EXECUTOR = None


def _get_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        timeout = int(os.environ.get('CODE_REWARD_TIMEOUT_SEC', '4'))
        memory_limit_mb = int(os.environ.get('CODE_REWARD_MEMORY_MB', '1024'))
        executor_cls = _load_executor_cls()
        _EXECUTOR = executor_cls(timeout=timeout, memory_limit_mb=memory_limit_mb)
    return _EXECUTOR


class CodeGen1PassRateReward(ORM):

    def __call__(self, completions, test_cases=None, fn_mode=None, **kwargs) -> List[float]:
        n = len(completions)
        test_cases_list = _ensure_list(test_cases, n)
        fn_mode_list = _ensure_list(fn_mode, n)

        executor = _get_executor()
        rewards: List[float] = []
        for completion, raw_test_cases, mode in zip(completions, test_cases_list, fn_mode_list):
            parsed_test_cases = _parse_test_cases(raw_test_cases)
            if parsed_test_cases is None:
                rewards.append(0.0)
                continue

            try:
                results = executor.evaluate(
                    model_response=completion,
                    test_samples=parsed_test_cases,
                    mode=(mode or 'auto'),
                )
            except Exception:
                rewards.append(0.0)
                continue

            if not results:
                rewards.append(0.0)
                continue

            passed_count = sum(1 for item in results if item.get('passed', False))
            rewards.append(float(passed_count) / float(len(results)))
        return rewards


orms['external_codegen1_pass_rate'] = CodeGen1PassRateReward


if __name__ == '__main__':
    dataset = load_dataset('codegen1_train_rl:rl#3', use_hf=True)[0]
    print(dataset)
