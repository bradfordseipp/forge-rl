"""Quick A/B benchmark: SyncVectorEnv vs ThreadedVecEnv."""

import time
import sys
import os
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))

from train_ppo import make_env, start_servers, OBS_KEYS
from forge_rl.threaded_vec_env import ThreadedVecEnv

NUM_ENVS = 16
NUM_STEPS = 50  # rollout steps to benchmark
BASE_PORT = 50051
DECK_A = "src/main/resources/decks/mono_red_pingers.dck"
DECK_B = "src/main/resources/decks/mono_red_pingers.dck"
SERVER_DIR = str(os.path.join(os.path.dirname(__file__), "..", "forge-research"))

print(f"Starting {NUM_ENVS} servers...")
procs = start_servers(NUM_ENVS, BASE_PORT, SERVER_DIR, DECK_A, DECK_B)
print(f"Servers started, waiting for readiness...")
time.sleep(15)

env_fns = [
    make_env(BASE_PORT + i, DECK_A, DECK_B, seed=42 + i, no_reward_shaping=True)
    for i in range(NUM_ENVS)
]


def bench(envs, label: str):
    obs, _ = envs.reset(seed=42)
    total_steps = 0
    t0 = time.time()
    for step in range(NUM_STEPS):
        # Random valid actions (fallback to 0 if mask is empty after auto-reset)
        masks = obs["action_mask"]
        actions = np.array([
            np.random.choice(np.where(masks[i] > 0)[0]) if masks[i].any() else 0
            for i in range(NUM_ENVS)
        ])
        obs, rewards, terminated, truncated, infos = envs.step(actions)
        total_steps += NUM_ENVS
    elapsed = time.time() - t0
    sps = total_steps / elapsed
    print(f"{label}: {total_steps} steps in {elapsed:.2f}s = {sps:.0f} SPS")
    return sps


# Benchmark SyncVectorEnv
print("\n--- SyncVectorEnv ---")
sync_envs = gym.vector.SyncVectorEnv([
    make_env(BASE_PORT + i, DECK_A, DECK_B, seed=42 + i, no_reward_shaping=True)
    for i in range(NUM_ENVS)
])
sync_sps = bench(sync_envs, "Sync")
sync_envs.close()

# Benchmark ThreadedVecEnv (reuse same servers)
print("\n--- ThreadedVecEnv ---")
threaded_envs = ThreadedVecEnv([
    make_env(BASE_PORT + i, DECK_A, DECK_B, seed=42 + i, no_reward_shaping=True)
    for i in range(NUM_ENVS)
])
threaded_sps = bench(threaded_envs, "Threaded")
threaded_envs.close()

print(f"\nSpeedup: {threaded_sps / sync_sps:.2f}x")

# Cleanup
for p in procs:
    p.kill()
print("Done.")
