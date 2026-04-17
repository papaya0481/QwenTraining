"""VeRPO-aware reward manager.

The key difference from DAPORewardManager: instead of calling compute_score
per-sample independently, this manager:

1. Runs code execution for all N samples in a prompt group in parallel.
2. Collects the per-testcase passed_flags for every sample in the group.
3. Computes group-level rho_j (eq. 1 in the paper) from those flags.
4. Computes R^turn = sum_j w_j' * p_j (eq. 4) for each sample using the
   shared group-level weights — exactly as the paper specifies.
5. Computes traj_reward with turn-count-based efficiency decay (eq. 5).

This manager is registered as "verpo" and is selected via:
    reward.reward_manager.name: verpo
in the Hydra config.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from scripts.plugins.verl_codegen1_reward import execute_single


def _default_reward_kwargs() -> dict[str, Any]:
    return {
        "timeout_sec": 4,
        "memory_limit_mb": 1024,
        "difficulty_alpha": 2.0,
        "density_eps": 1e-6,
        "density_sigma_floor": 1e-3,
        "efficiency_gamma": 0.995,
        "efficiency_mode": "turn_count",
    }


@register("verpo")
class VeRPORewardManager(AbstractRewardManager):
    """Reward manager that computes group-level rho_j per the VeRPO paper."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,          # unused — kept for interface compat
        reward_fn_key: str = "data_source",
        max_resp_len: int | None = None,
        overlong_buffer_cfg=None,
        reward_kwargs: dict[str, Any] | None = None,
        num_workers: int = 4,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.num_workers = num_workers

        rk = _default_reward_kwargs()
        if reward_kwargs:
            rk.update(reward_kwargs)
        self.reward_kwargs = rk

        if overlong_buffer_cfg is not None:
            assert max_resp_len is not None
            assert max_resp_len >= overlong_buffer_cfg.len
            assert not overlong_buffer_cfg.enable or overlong_buffer_cfg.len > 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _decode_response(self, data_item) -> tuple[str, str]:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        eos = self.tokenizer.eos_token
        if eos and response_str.endswith(eos):
            response_str = response_str[: -len(eos)]
        return prompt_str, response_str

    def _valid_response_length(self, data_item) -> int:
        prompt_length = data_item.batch["prompts"].shape[-1]
        return int(data_item.batch["attention_mask"][prompt_length:].sum())

    # ------------------------------------------------------------------
    # main __call__
    # ------------------------------------------------------------------

    def __call__(self, data: DataProto, return_dict: bool = False):
        from verl.workers.reward_manager.abstract import AbstractRewardManager
        reward_from_rm = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm is not None:
            return reward_from_rm

        n = len(data)
        rk = self.reward_kwargs

        # ---- Step 1: decode + execute all samples in parallel ----
        raw_results: list[dict[str, Any] | None] = [None] * n
        response_strs: list[str] = [""] * n
        prompt_strs: list[str] = [""] * n

        def _run(i):
            item = data[i]
            p_str, r_str = self._decode_response(item)
            gt = item.non_tensor_batch["reward_model"]["ground_truth"]
            extra = dict(item.non_tensor_batch.get("extra_info") or {})
            extra["rollout_reward_scores"] = item.non_tensor_batch.get("reward_scores", {})
            raw = execute_single(
                solution_str=r_str,
                ground_truth=gt,
                extra_info=extra,
                timeout_sec=rk["timeout_sec"],
                memory_limit_mb=rk["memory_limit_mb"],
            )
            return i, p_str, r_str, raw

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(_run, i) for i in range(n)]
            for fut in as_completed(futures):
                i, p_str, r_str, raw = fut.result()
                raw_results[i] = raw
                response_strs[i] = r_str
                prompt_strs[i] = p_str

        # ---- Step 2: group samples by uid (= prompt group) ----
        # uid is set by the trainer before reward computation
        uids: list[str] = list(data.non_tensor_batch.get("uid", [str(i) for i in range(n)]))
        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, uid in enumerate(uids):
            uid_to_indices[uid].append(i)

        # ---- Step 3: for each group, compute group-level rho_j then score ----
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)
        # pre-fill with zeros so indices align even for failed samples
        for key in ("score", "acc", "passed", "total", "pass_rate",
                    "dense_reward", "traj_reward", "outcome_reward",
                    "efficiency_decay", "avg_difficulty_weight", "density_sigma"):
            reward_extra_info[key] = [0.0] * n

        already_print: dict[str, int] = {}

        for uid, indices in uid_to_indices.items():
            valid_indices = [i for i in indices if raw_results[i] is not None]
            group_flags = [raw_results[i]["passed_flags"] for i in valid_indices]

            # Vectorize group-level weights once per group
            if group_flags:
                group_arr = np.array(group_flags, dtype=np.float64)      # [N, M]
                rho = group_arr.mean(axis=0)                              # [M]
                base_w = np.exp(-rk["difficulty_alpha"] * rho)            # [M]
                sigma_val = float(max(rho.std() / 2.0, rk["density_sigma_floor"]))
                diff = rho[:, None] - rho[None, :]                        # [M, M]
                density = np.exp(
                    -(diff ** 2) / (2.0 * sigma_val * sigma_val + rk["density_eps"])
                ).sum(axis=1)
                norm_w = base_w / (density + rk["density_eps"])           # [M]
                avg_w_group = float(base_w.mean())
            else:
                norm_w = avg_w_group = sigma_val = None

            for i in indices:
                raw = raw_results[i]
                valid_resp_len = self._valid_response_length(data[i])
                data_source = data[i].non_tensor_batch[self.reward_fn_key]

                if raw is None:
                    reward_tensor[i, max(valid_resp_len - 1, 0)] = 0.0
                    continue

                if norm_w is not None:
                    q = np.asarray(raw["passed_flags"], dtype=np.float64)
                    dense_reward = float(np.dot(norm_w, q))
                    avg_w, sigma = avg_w_group, sigma_val
                else:
                    dense_reward, avg_w, sigma = 0.0, 0.0, rk["density_sigma_floor"]

                # Efficiency decay — paper uses turn count (|tau|)
                efficiency_decay = 1.0
                mode = rk["efficiency_mode"]
                gamma = rk["efficiency_gamma"]
                if mode == "turn_count":
                    efficiency_decay = gamma ** max(raw["turn_count"], 1)
                elif mode == "response_tokens":
                    tok = max(len(response_strs[i].split()), 1)
                    efficiency_decay = gamma ** max(tok - 1, 0)

                traj_reward = raw["outcome_reward"] * efficiency_decay

                # score = traj_reward (used by reward_tensor path)
                score = traj_reward
                reward = score

                # Overlong penalty
                if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                    buf_len = self.overlong_buffer_cfg.len
                    expected = self.max_resp_len - buf_len
                    exceed = valid_resp_len - expected
                    penalty = min(-exceed / buf_len * self.overlong_buffer_cfg.penalty_factor, 0)
                    reward += penalty
                    if self.overlong_buffer_cfg.log:
                        reward_extra_info["overlong_reward"][i] = penalty
                        reward_extra_info["overlong"][i] = penalty < 0

                reward_tensor[i, max(valid_resp_len - 1, 0)] = reward

                reward_extra_info["score"][i] = score
                reward_extra_info["acc"][i] = raw["pass_rate"]
                reward_extra_info["passed"][i] = raw["passed"]
                reward_extra_info["total"][i] = raw["total"]
                reward_extra_info["pass_rate"][i] = raw["pass_rate"]
                reward_extra_info["dense_reward"][i] = dense_reward
                reward_extra_info["traj_reward"][i] = traj_reward
                reward_extra_info["outcome_reward"][i] = raw["outcome_reward"]
                reward_extra_info["efficiency_decay"][i] = efficiency_decay
                reward_extra_info["avg_difficulty_weight"][i] = avg_w
                reward_extra_info["density_sigma"][i] = sigma

                # Console logging
                if data_source not in already_print:
                    already_print[data_source] = 0
                if already_print[data_source] < self.num_examine:
                    already_print[data_source] += 1
                    print("[prompt]", prompt_strs[i])
                    print("[response]", response_strs[i])
                    print("[dense_reward]", dense_reward, "[traj_reward]", traj_reward,
                          "[pass_rate]", raw["pass_rate"])

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
