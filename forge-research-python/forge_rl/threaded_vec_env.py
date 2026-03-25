"""Threaded vector environment for gRPC-backed Forge envs.

Uses ThreadPoolExecutor to parallelize gRPC step/reset calls across envs.
gRPC releases the GIL during network I/O, so threads achieve true parallelism
for the blocking RPC calls without subprocess overhead.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import gymnasium as gym
import numpy as np


class ThreadedVecEnv:
    """Drop-in replacement for gym.vector.SyncVectorEnv using threads.

    Matches the subset of the VectorEnv API used by train_ppo.py:
      - single_observation_space, single_action_space, num_envs
      - reset(seed=...) -> (obs_dict, info_dict)
      - step(actions) -> (obs_dict, rewards, terminated, truncated, infos)
      - close()

    Individual envs should be wrapped with RecordEpisodeStatistics.
    This class batches their episode info into the _episode mask format
    that Gymnasium's vectorized envs use.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]]):
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs)

        # Mirror gymnasium VectorEnv interface
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

        # Auto-reset tracking: which envs need reset on next step
        self._autoreset = np.zeros(self.num_envs, dtype=bool)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset all envs in parallel."""
        def _reset(i):
            env_seed = (seed + i) if seed is not None else None
            return self.envs[i].reset(seed=env_seed, options=options)

        futures = [self._executor.submit(_reset, i) for i in range(self.num_envs)]
        results = [f.result() for f in futures]

        obs_list = [r[0] for r in results]
        self._autoreset[:] = False

        return self._batch_obs(obs_list), {}

    def step(self, actions: np.ndarray):
        """Step all envs in parallel with threaded gRPC calls."""
        def _step_or_reset(i):
            if self._autoreset[i]:
                # This env terminated last step — reset first, then step
                obs, info = self.envs[i].reset()
                return ("reset", obs, info)
            else:
                obs, reward, terminated, truncated, info = self.envs[i].step(int(actions[i]))
                return ("step", obs, reward, terminated, truncated, info)

        futures = [self._executor.submit(_step_or_reset, i) for i in range(self.num_envs)]
        results = [f.result() for f in futures]

        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terminated_arr = np.zeros(self.num_envs, dtype=bool)
        truncated_arr = np.zeros(self.num_envs, dtype=bool)
        info_list = []

        for i, result in enumerate(results):
            if result[0] == "reset":
                _, obs, info = result
                obs_list.append(obs)
                info_list.append(info)
                self._autoreset[i] = False
            else:
                _, obs, r, term, trunc, info = result
                obs_list.append(obs)
                rewards[i] = r
                terminated_arr[i] = term
                truncated_arr[i] = trunc
                info_list.append(info)

                if term or trunc:
                    self._autoreset[i] = True

        batched_obs = self._batch_obs(obs_list)
        batched_infos = self._batch_infos(info_list)

        # Batch episode stats from RecordEpisodeStatistics wrappers
        # into the _episode mask format that train_ppo expects
        finished_mask = np.zeros(self.num_envs, dtype=bool)
        ep_returns = np.zeros(self.num_envs, dtype=np.float64)
        ep_lengths = np.zeros(self.num_envs, dtype=np.int64)
        for i, info in enumerate(info_list):
            if "episode" in info:
                finished_mask[i] = True
                ep_returns[i] = info["episode"]["r"]
                ep_lengths[i] = info["episode"]["l"]

        if finished_mask.any():
            batched_infos["episode"] = {"r": ep_returns, "l": ep_lengths}
            batched_infos["_episode"] = finished_mask

        return batched_obs, rewards, terminated_arr, truncated_arr, batched_infos

    def close(self):
        """Close all envs and shutdown thread pool."""
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
        self._executor.shutdown(wait=False)

    def _batch_obs(self, obs_list: list[dict]) -> dict:
        """Stack per-env obs dicts into batched numpy arrays."""
        keys = obs_list[0].keys()
        return {
            key: np.stack([obs[key] for obs in obs_list])
            for key in keys
        }

    def _batch_infos(self, info_list: list[dict]) -> dict:
        """Batch info dicts, handling game_result specially."""
        batched = {}
        has_game_result = [("game_result" in info) for info in info_list]
        if any(has_game_result):
            gr_mask = np.array(has_game_result, dtype=bool)
            batched["_game_result"] = gr_mask
            batched["game_result"] = {
                "winner_index": np.array([
                    info.get("game_result", {}).get("winner_index", -1)
                    for info in info_list
                ]),
                "is_draw": np.array([
                    info.get("game_result", {}).get("is_draw", False)
                    for info in info_list
                ]),
                "turns_played": np.array([
                    info.get("game_result", {}).get("turns_played", 0)
                    for info in info_list
                ]),
                "win_condition": np.array([
                    info.get("game_result", {}).get("win_condition", "")
                    for info in info_list
                ]),
            }
        return batched
