"""CleanRL-style PPO training script for Forge RL with action masking."""

import atexit
import os
import random
import socket
import subprocess
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from forge_rl.env import ForgeRlEnv


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class Args:
    # Environment
    num_envs: int = 8
    num_steps: int = 128
    total_timesteps: int = 5_000_000
    deck_a: str = "src/main/resources/decks/mono_red_pingers.dck"
    deck_b: str = "src/main/resources/decks/caw_gates.dck"

    # Server
    server_base_port: int = 50051
    server_jar_dir: str = str(Path(__file__).resolve().parent.parent / "forge-research")

    # PPO
    learning_rate: float = 2.5e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Misc
    seed: int = 1
    track: bool = False
    run_name: str = ""
    anneal_lr: bool = True

    # Dual RL (train against frozen opponent)
    opponent_model: str = ""
    swap_decks: bool = False
    no_reward_shaping: bool = False
    resume: str = ""

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


# ---------------------------------------------------------------------------
# Masked Categorical
# ---------------------------------------------------------------------------

class CategoricalMasked(Categorical):
    """Categorical distribution with invalid-action masking."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        self.mask = mask.bool()
        masked_logits = logits.clone()
        masked_logits[~self.mask] = -1e8
        super().__init__(logits=masked_logits)

    def entropy(self):
        # Compute entropy only over valid actions, normalized to [0, 1]
        p_log_p = self.logits * self.probs
        p_log_p[~self.mask] = 0.0
        raw = -p_log_p.sum(dim=-1)
        max_ent = torch.log(self.mask.sum(dim=-1).clamp(min=2).float())
        return raw / max_ent


# ---------------------------------------------------------------------------
# Agent (Policy + Value Network)
# ---------------------------------------------------------------------------

CARD_ZONES = [
    "agent_hand", "agent_battlefield", "opponent_battlefield",
    "agent_graveyard", "opponent_graveyard",
    "agent_exile", "opponent_exile",
]
CARD_FEATURES = 38  # 12 scalar + 5 colors + 7 types + 14 keywords
STACK_FEATURES = 2  # source_name_id, controller_index (0=agent, 1=opponent)
# Zone sizes (must match env.py)
ZONE_SIZES = {"agent_hand": 10, "agent_battlefield": 20, "opponent_battlefield": 20,
              "agent_graveyard": 15, "opponent_graveyard": 15, "agent_exile": 10, "opponent_exile": 10}
TOTAL_CARD_SLOTS = sum(ZONE_SIZES.values())  # 100
CARD_VOCAB_SIZE = 4096  # max unique card names (name_id 0 reserved for padding)
CARD_EMBED_DIM = 64     # learned embedding dimension per card name
NUM_ZONES = len(CARD_ZONES)  # 7
CARD_TOKEN_DIM = 64     # per-card token dimension fed into attention
ATTN_HEADS = 4
ATTN_LAYERS = 2


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        # Scalar encoder: game_info(6) + agent_scalars(11) + opponent_scalars(11) + decision_type(15) = 43
        scalar_dim = 64
        self.scalar_encoder = nn.Sequential(
            layer_init(nn.Linear(43, scalar_dim)),
            nn.ReLU(),
        )

        # Card name embedding (name_id → learned vector)
        self.card_name_embedding = nn.Embedding(CARD_VOCAB_SIZE, CARD_EMBED_DIM, padding_idx=0)

        # Zone embedding: learned vector per zone, added to each card token
        self.zone_embedding = nn.Embedding(NUM_ZONES, CARD_TOKEN_DIM)

        # Card projection: raw features → token dim for attention
        card_input_dim = CARD_EMBED_DIM + CARD_FEATURES - 1  # 64 + 37 = 101
        self.card_proj = nn.Sequential(
            layer_init(nn.Linear(card_input_dim, CARD_TOKEN_DIM)),
            nn.ReLU(),
        )

        # Cross-zone transformer: self-attention over all card tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CARD_TOKEN_DIM,
            nhead=ATTN_HEADS,
            dim_feedforward=CARD_TOKEN_DIM * 4,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.card_transformer = nn.TransformerEncoder(encoder_layer, num_layers=ATTN_LAYERS)

        # Stack encoder
        self.stack_encoder = nn.Sequential(
            layer_init(nn.Linear(STACK_FEATURES, 16)),
            nn.ReLU(),
        )

        # --- Actor: actions cross-attend to card tokens ---
        # Action projection: raw action features → same dim as card tokens
        action_proj_input = 2 * CARD_EMBED_DIM + 32  # src_emb(64) + tgt_emb(64) + 32 other = 160
        self.action_proj = nn.Sequential(
            layer_init(nn.Linear(action_proj_input, CARD_TOKEN_DIM)),
            nn.ReLU(),
        )

        # Cross-attention: action tokens (queries) attend to card tokens (keys/values)
        self.action_card_cross_attn = nn.MultiheadAttention(
            embed_dim=CARD_TOKEN_DIM,
            num_heads=ATTN_HEADS,
            dropout=0.0,
            batch_first=True,
        )
        self.action_cross_attn_norm = nn.LayerNorm(CARD_TOKEN_DIM)

        # Action scoring head: cross_attn_output(64) + scalar(64) + stack(16) = 144 → 1
        context_dim = scalar_dim + 16  # 80
        self.action_score = nn.Sequential(
            layer_init(nn.Linear(CARD_TOKEN_DIM + context_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=0.01),
        )

        # --- Critic: pooled board state → value ---
        card_pool_dim = 128
        self.card_pool = nn.Sequential(
            layer_init(nn.Linear(CARD_TOKEN_DIM, card_pool_dim)),
            nn.ReLU(),
        )
        critic_trunk_input = scalar_dim + card_pool_dim + 16  # 64 + 128 + 16 = 208
        self.critic_trunk = nn.Sequential(
            layer_init(nn.Linear(critic_trunk_input, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
        )
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)

    def _tokenize_cards(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build card tokens from all zones with zone embeddings.

        Returns:
            tokens: (batch, TOTAL_CARD_SLOTS, CARD_TOKEN_DIM)
            key_padding_mask: (batch, TOTAL_CARD_SLOTS) — True for padding slots
        """
        all_tokens = []
        all_masks = []

        for zone_idx, zone in enumerate(CARD_ZONES):
            card_matrix = obs[zone]  # (batch, zone_size, CARD_FEATURES)

            # Padding mask: cards with first feature < 0 are empty
            is_padding = card_matrix[:, :, 0] < 0  # (batch, zone_size)

            # Embed name_id and project features
            name_ids = card_matrix[:, :, 0].long().clamp(min=0, max=CARD_VOCAB_SIZE - 1)
            name_emb = self.card_name_embedding(name_ids)
            other_features = card_matrix[:, :, 1:].float()
            card_input = torch.cat([name_emb, other_features], dim=-1)
            tokens = self.card_proj(card_input)  # (batch, zone_size, CARD_TOKEN_DIM)

            # Add zone embedding
            tokens = tokens + self.zone_embedding.weight[zone_idx]

            # Zero out padding tokens
            tokens = tokens * (~is_padding).float().unsqueeze(-1)

            all_tokens.append(tokens)
            all_masks.append(is_padding)

        return torch.cat(all_tokens, dim=1), torch.cat(all_masks, dim=1)

    def _encode_stack(self, stack_matrix: torch.Tensor) -> torch.Tensor:
        """Encode stack entries with mean pooling."""
        valid = (stack_matrix[:, :, 0] >= 0).float().unsqueeze(-1)
        encoded = self.stack_encoder(stack_matrix.float())
        encoded = encoded * valid
        count = valid.sum(dim=1).clamp(min=1)
        pooled = encoded.sum(dim=1) / count
        return pooled

    def _normalize_scalars(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize and concatenate scalar observations."""
        gi = obs["game_info"].float()
        gi_norm = torch.stack([
            gi[:, 0] / 20.0, gi[:, 1] / 18.0, gi[:, 2], gi[:, 3],
            gi[:, 4] / 5.0, gi[:, 5] / 5.0,
        ], dim=-1)
        a_sc = obs["agent_scalars"].float()
        a_norm = torch.stack([
            a_sc[:, 0] / 20.0, a_sc[:, 1] / 7.0, a_sc[:, 2] / 60.0,
            a_sc[:, 3] / 5.0, a_sc[:, 4] / 5.0,
            a_sc[:, 5] / 5.0, a_sc[:, 6] / 5.0, a_sc[:, 7] / 5.0,
            a_sc[:, 8] / 5.0, a_sc[:, 9] / 5.0, a_sc[:, 10] / 5.0,
        ], dim=-1)
        o_sc = obs["opponent_scalars"].float()
        o_norm = torch.stack([
            o_sc[:, 0] / 20.0, o_sc[:, 1] / 7.0, o_sc[:, 2] / 60.0,
            o_sc[:, 3] / 5.0, o_sc[:, 4] / 5.0,
            o_sc[:, 5] / 5.0, o_sc[:, 6] / 5.0, o_sc[:, 7] / 5.0,
            o_sc[:, 8] / 5.0, o_sc[:, 9] / 5.0, o_sc[:, 10] / 5.0,
        ], dim=-1)
        return torch.cat([gi_norm, a_norm, o_norm, obs["decision_type"].float()], dim=-1)

    def _encode_board(self, obs: dict[str, torch.Tensor]):
        """Encode the full board state. Returns scalar_enc, card_tokens, card_mask, stack_enc."""
        scalar_enc = self.scalar_encoder(self._normalize_scalars(obs))  # (B, 64)

        card_tokens, card_mask = self._tokenize_cards(obs)  # (B, 100, 64), (B, 100)
        card_tokens = self.card_transformer(card_tokens, src_key_padding_mask=card_mask)

        stack_enc = self._encode_stack(obs["stack"])  # (B, 16)

        return scalar_enc, card_tokens, card_mask, stack_enc

    def _encode_actions(self, action_features: torch.Tensor) -> torch.Tensor:
        """Encode raw action features into action tokens.

        action_features: (B, 256, 34) where indices 0 and 4 are raw name_ids.
        Returns: (B, 256, CARD_TOKEN_DIM) action tokens.
        """
        af = action_features.float()
        src_name_ids = af[:, :, 0].long().clamp(min=0, max=CARD_VOCAB_SIZE - 1)
        tgt_name_ids = af[:, :, 4].long().clamp(min=0, max=CARD_VOCAB_SIZE - 1)
        src_emb = self.card_name_embedding(src_name_ids)  # (B, 256, 64)
        tgt_emb = self.card_name_embedding(tgt_name_ids)  # (B, 256, 64)

        other = torch.cat([af[:, :, 1:4], af[:, :, 5:7], af[:, :, 7:34]], dim=-1)  # (B, 256, 32)

        combined = torch.cat([src_emb, other, tgt_emb], dim=-1)  # (B, 256, 160)
        return self.action_proj(combined)  # (B, 256, 64)

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        scalar_enc, card_tokens, card_mask, stack_enc = self._encode_board(obs)

        # Mean pool card tokens for critic
        valid = (~card_mask).float().unsqueeze(-1)
        pooled = (card_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        card_enc = self.card_pool(pooled)  # (B, 128)

        combined = torch.cat([scalar_enc, card_enc, stack_enc], dim=-1)  # (B, 208)
        return self.critic(self.critic_trunk(combined))

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
    ):
        scalar_enc, card_tokens, card_mask, stack_enc = self._encode_board(obs)

        # --- Actor: actions cross-attend to board ---
        action_tokens = self._encode_actions(obs["action_features"])  # (B, 256, 64)

        # Cross-attention: actions query the card tokens
        # Ensure at least one key is unmasked to prevent NaN from all-masked softmax
        safe_mask = card_mask.clone()
        all_masked = card_mask.all(dim=1)  # (B,)
        if all_masked.any():
            safe_mask[all_masked, 0] = False
        attn_out, _ = self.action_card_cross_attn(
            query=action_tokens,
            key=card_tokens,
            value=card_tokens,
            key_padding_mask=safe_mask,
        )  # (B, 256, 64)
        action_repr = self.action_cross_attn_norm(action_tokens + attn_out)  # residual + norm

        # Broadcast scalar context to each action and score
        context = torch.cat([scalar_enc, stack_enc], dim=-1)  # (B, 80)
        context_expanded = context.unsqueeze(1).expand(-1, 256, -1)  # (B, 256, 80)
        score_input = torch.cat([action_repr, context_expanded], dim=-1)  # (B, 256, 144)
        logits = self.action_score(score_input).squeeze(-1)  # (B, 256)

        # --- Critic: pooled board → value ---
        valid = (~card_mask).float().unsqueeze(-1)
        pooled = (card_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        card_enc = self.card_pool(pooled)  # (B, 128)
        critic_input = torch.cat([scalar_enc, card_enc, stack_enc], dim=-1)  # (B, 208)
        value = self.critic(self.critic_trunk(critic_input))

        mask = obs["action_mask"]
        dist = CategoricalMasked(logits, mask)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


# ---------------------------------------------------------------------------
# Observation utilities
# ---------------------------------------------------------------------------

OBS_KEYS = [
    "game_info", "agent_scalars", "opponent_scalars",
    "agent_hand", "agent_battlefield", "opponent_battlefield",
    "agent_graveyard", "opponent_graveyard",
    "agent_exile", "opponent_exile",
    "stack", "action_mask",
    "decision_type", "action_features",
]


DECISION_TYPE_NAMES = [
    "choose_spell_ability", "declare_attackers", "declare_blockers",
    "choose_targets", "mulligan", "choose_cards", "confirm_action",
    "choose_entity", "choose_color", "choose_number", "choose_type",
    "choose_pile", "choose_mode", "order_cards", "ai_fallback",
]
DT_MULLIGAN = 4

# Card feature indices (matching env.py unpacked layout)
CARD_IDX_NAME_ID = 0
CARD_IDX_CMC = 3
CARD_IDX_IS_LAND = 17  # unpacked from type_bitmask


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert a dict of numpy arrays to a dict of torch tensors."""
    return {k: torch.tensor(obs[k], device=device) for k in OBS_KEYS}



# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def _port_is_open(port: int, host: str = "localhost", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def _build_classpath(server_dir: str) -> str:
    """Build the Java classpath from Maven output."""
    cp_file = os.path.join(server_dir, "target", "classpath.txt")
    if not os.path.exists(cp_file):
        raise FileNotFoundError(
            f"Classpath file not found: {cp_file}\n"
            "Run 'mvn dependency:build-classpath -Dmdep.outputFile=target/classpath.txt' in forge-research/"
        )
    with open(cp_file, "r") as f:
        deps_cp = f.read().strip()

    # Add module target/classes dirs
    module_dirs = [
        os.path.join(server_dir, "target", "classes"),
    ]
    # Also add sibling module classes that might be needed
    forge_root = os.path.dirname(server_dir)
    for module in ["forge-core", "forge-game", "forge-ai", "forge-gui", "forge-gui-desktop"]:
        classes_dir = os.path.join(forge_root, module, "target", "classes")
        if os.path.isdir(classes_dir):
            module_dirs.append(classes_dir)

    return os.pathsep.join(module_dirs) + os.pathsep + deps_cp


def start_servers(
    num_servers: int,
    base_port: int,
    server_dir: str,
    deck_paths: list[str] | None = None,
    log_dir: str | None = None,
) -> list[subprocess.Popen]:
    """Spawn N ForgeResearchServer Java processes sequentially.

    When deck_paths are provided, enables lazy card loading — each server
    only loads the cards from those decks (~30-50MB vs ~400MB full DB).
    When log_dir is provided, server stdout/stderr is written to per-server log files.
    """
    classpath = _build_classpath(server_dir)
    procs = []

    # Resolve deck paths to absolute — deck paths are relative to server_dir
    # (e.g. src/main/resources/decks/mono_red.dck is under forge-research/)
    abs_deck_paths = []
    if deck_paths:
        for dp in deck_paths:
            p = Path(dp)
            if not p.is_absolute():
                p = Path(server_dir) / dp
            abs_deck_paths.append(str(p.resolve()))

    # Set up log directory
    log_files = []
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_servers):
        port = base_port + i
        if _port_is_open(port):
            raise RuntimeError(f"Port {port} is already in use")

        cmd = ["java", "-Xmx128m", "-cp", classpath,
               "forge.research.ForgeResearchServer", str(port)]
        cmd.extend(abs_deck_paths)

        if log_dir:
            log_path = Path(log_dir) / f"server_{port}.log"
            log_file = open(log_path, "w")
            log_files.append(log_file)
            stdout_dest = log_file
            stderr_dest = log_file
        else:
            stdout_dest = subprocess.DEVNULL
            stderr_dest = subprocess.DEVNULL

        proc = subprocess.Popen(
            cmd,
            cwd=server_dir,
            stdout=stdout_dest,
            stderr=stderr_dest,
        )
        procs.append(proc)
        print(f"  Starting server {i+1}/{num_servers} on port {port} (PID={proc.pid})...")

        # Wait for this server before launching the next
        for attempt in range(120):  # up to 60 seconds
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Server on port {port} exited with code {proc.returncode}"
                )
            if _port_is_open(port):
                print(f"  Server on port {port} is ready")
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Server on port {port} did not start within 60 seconds")

    return procs


def stop_servers(procs: list[subprocess.Popen]):
    """Terminate all server processes."""
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("All servers stopped.")


# ---------------------------------------------------------------------------
# Reward shaping wrapper
# ---------------------------------------------------------------------------

class RewardShaping(gym.Wrapper):
    """Add intermediate rewards to guide learning.

    - Opponent life change: +0.05 per point of damage dealt, -0.05 per point gained.
      Applied per decision step (damage events happen at specific moments).
    - Board creature presence: +0.01 per creature per turn (not per decision step).
      Only counts creatures (cards with power >= 0, since lands/artifacts have power = -1).
    - Terminal +1/-1 from the base env is preserved unchanged.
    """

    LIFE_SCALE = 0.005   # per opponent life point lost
    BOARD_SCALE = 0.01   # per creature per turn (applied once when turn changes)

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_opp_life = 20
        self._prev_turn = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_opp_life = int(obs["opponent_scalars"][0])
        self._prev_turn = int(obs["game_info"][0])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not terminated:
            opp_life = int(obs["opponent_scalars"][0])

            # Reward for damage dealt, penalty for opponent lifegain
            opp_life_lost = self._prev_opp_life - opp_life
            reward += self.LIFE_SCALE * opp_life_lost
            self._prev_opp_life = opp_life

            # Board creature presence: only on turn change to avoid per-decision accumulation
            current_turn = int(obs["game_info"][0])
            if current_turn != self._prev_turn:
                agent_bf = obs["agent_battlefield"]
                # Count creatures using the is_creature type bit (index 16)
                num_creatures = int((agent_bf[:, 16] > 0).sum())
                reward += self.BOARD_SCALE * num_creatures
                self._prev_turn = current_turn

        return obs, reward, terminated, truncated, info



# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(port: int, deck_a: str, deck_b: str, seed: int,
             agent_player_index: int = 0, opponent_model_path: str = "",
             no_reward_shaping: bool = False):
    """Create an env factory function for the given port."""
    def thunk():
        env = ForgeRlEnv(
            deck_path_a=deck_a,
            deck_path_b=deck_b,
            agent_player_index=agent_player_index,
            port=port,
            opponent_model_path=opponent_model_path if opponent_model_path else None,
        )
        if not no_reward_shaping:
            env = RewardShaping(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> Args:
    import argparse
    parser = argparse.ArgumentParser(description="PPO training for Forge RL")
    args_obj = Args()

    parser.add_argument("--num-envs", type=int, default=args_obj.num_envs)
    parser.add_argument("--num-steps", type=int, default=args_obj.num_steps)
    parser.add_argument("--total-timesteps", type=int, default=args_obj.total_timesteps)
    parser.add_argument("--deck-a", type=str, default=args_obj.deck_a)
    parser.add_argument("--deck-b", type=str, default=args_obj.deck_b)
    parser.add_argument("--server-base-port", type=int, default=args_obj.server_base_port)
    parser.add_argument("--server-jar-dir", type=str, default=args_obj.server_jar_dir)
    parser.add_argument("--learning-rate", type=float, default=args_obj.learning_rate)
    parser.add_argument("--gamma", type=float, default=args_obj.gamma)
    parser.add_argument("--gae-lambda", type=float, default=args_obj.gae_lambda)
    parser.add_argument("--num-minibatches", type=int, default=args_obj.num_minibatches)
    parser.add_argument("--update-epochs", type=int, default=args_obj.update_epochs)
    parser.add_argument("--clip-coef", type=float, default=args_obj.clip_coef)
    parser.add_argument("--ent-coef", type=float, default=args_obj.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=args_obj.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=args_obj.max_grad_norm)
    parser.add_argument("--seed", type=int, default=args_obj.seed)
    parser.add_argument("--track", action="store_true", default=args_obj.track)
    parser.add_argument("--run-name", type=str, default=args_obj.run_name)
    parser.add_argument("--anneal-lr", type=lambda x: x.lower() == "true", default=args_obj.anneal_lr)
    parser.add_argument("--opponent-model", type=str, default=args_obj.opponent_model,
                        help="Path to frozen .onnx model for opponent")
    parser.add_argument("--swap-decks", action="store_true", default=args_obj.swap_decks,
                        help="Agent plays deck B instead of A (player index 1)")
    parser.add_argument("--no-reward-shaping", action="store_true", default=False,
                        help="Disable intermediate reward shaping (pure win/loss)")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint .pt file to resume training from")
    parsed = parser.parse_args()
    return Args(**vars(parsed))


def main():
    args = parse_args()

    if not args.run_name:
        runs_dir = Path("runs")
        existing = sorted(runs_dir.glob("run_*")) if runs_dir.exists() else []
        next_num = 1
        for p in existing:
            try:
                n = int(p.name.split("_")[1])
                next_num = max(next_num, n + 1)
            except (IndexError, ValueError):
                pass
        args.run_name = f"run_{next_num:03d}"

    num_updates = args.total_timesteps // args.batch_size

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Logging
    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(
        f"|{k}|{v}|" for k, v in vars(args).items()
    ))

    if args.track:
        import wandb
        wandb.init(project="forge-rl", name=args.run_name, config=vars(args), save_code=True)

    # Opponent model path (ONNX) — passed to Java server for local inference
    opponent_model_path = ""
    agent_player_index = 0
    if args.opponent_model:
        opponent_model_path = str(Path(args.opponent_model).resolve())
        print(f"Using ONNX opponent model: {opponent_model_path}")
        if args.swap_decks:
            agent_player_index = 1
            print("Agent plays deck B (player index 1)")

    # Start servers (pass deck paths for lazy card loading)
    log_dir = f"runs/{args.run_name}/server_logs"
    print(f"Starting {args.num_envs} game servers (logs: {log_dir})...")
    server_procs = start_servers(args.num_envs, args.server_base_port, args.server_jar_dir,
                                 deck_paths=[args.deck_a, args.deck_b], log_dir=log_dir)
    atexit.register(stop_servers, server_procs)

    # Create vectorized environment
    envs = gym.vector.SyncVectorEnv([
        make_env(args.server_base_port + i, args.deck_a, args.deck_b, args.seed + i,
                 agent_player_index=agent_player_index, opponent_model_path=opponent_model_path,
                 no_reward_shaping=args.no_reward_shaping)
        for i in range(args.num_envs)
    ])

    # Agent
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # Resume from checkpoint
    start_update = 1
    resume_global_step = 0
    resume_episode_count = 0
    resume_win_count = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            agent.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_update = ckpt["update"] + 1
            resume_global_step = ckpt["global_step"]
            resume_episode_count = ckpt["episode_count"]
            resume_win_count = ckpt["win_count"]
            print(f"Resumed from {args.resume} (update {ckpt['update']}, step {resume_global_step:,})")
        else:
            # Legacy checkpoint (just state_dict)
            agent.load_state_dict(ckpt if isinstance(ckpt, dict) else ckpt)
            print(f"Loaded weights from {args.resume} (legacy format, starting from update 1)")

    # Rollout storage (numpy dicts for obs, tensors for the rest)
    obs_buf = {
        key: np.zeros((args.num_steps, args.num_envs) + envs.single_observation_space[key].shape,
                       dtype=envs.single_observation_space[key].dtype)
        for key in OBS_KEYS
    }
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Initial reset
    next_obs, info = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    # Tracking
    global_step = resume_global_step
    episode_count = resume_episode_count
    win_count = resume_win_count
    recent_returns = deque(maxlen=1000)
    recent_wins = deque(maxlen=1000)  # 1/0 per game, independent of reward shaping
    recent_turns = deque(maxlen=1000)
    recent_agent_life = deque(maxlen=1000)
    recent_opp_life = deque(maxlen=1000)

    # Decision type & mulligan tracking (reset each update)
    decision_type_counts = Counter()
    mulligan_events = []  # [(land_count, action_taken, mulligan_count), ...]

    print(f"Starting training: {num_updates} updates, {args.batch_size} batch size")
    start_time = time.time()

    for update in range(start_update, num_updates + 1):
        # Anneal learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            lr = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr

        # --- Rollout phase ---
        for step in range(args.num_steps):
            global_step += args.num_envs

            # Store obs and done
            for key in OBS_KEYS:
                obs_buf[key][step] = next_obs[key]
            dones[step] = next_done

            # Get action from policy
            with torch.no_grad():
                obs_t = obs_to_tensor(next_obs, device)
                action, logprob, _, value = agent.get_action_and_value(obs_t)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Track decision types and mulligan quality
            for env_i in range(args.num_envs):
                dt = int(next_obs["decision_type"][env_i].argmax())
                decision_type_counts[dt] += 1

                if dt == DT_MULLIGAN:
                    hand = next_obs["agent_hand"][env_i]  # (MAX_HAND, CARD_FEATURES)
                    occupied = hand[:, CARD_IDX_NAME_ID] > 0
                    is_land = occupied & (hand[:, CARD_IDX_IS_LAND] > 0)
                    land_count = int(is_land.sum())
                    mulligan_count = int(next_obs["game_info"][env_i][4])
                    action_taken = int(action[env_i].item())
                    mulligan_events.append((land_count, action_taken, mulligan_count))

            # Step environment
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)

            # Log completed episodes (Gymnasium v1.x uses "episode"/"_episode" keys)
            if "episode" in infos and "_episode" in infos:
                # _episode is a boolean mask indicating which envs finished
                finished_mask = infos["_episode"]
                for i, finished in enumerate(finished_mask):
                    if finished:
                        ep_return = float(infos["episode"]["r"][i])
                        ep_length = int(infos["episode"]["l"][i])
                        episode_count += 1
                        recent_returns.append(ep_return)
                        # Determine win from game_result if available, else from opp life
                        won = False
                        if "game_result" in infos and infos.get("_game_result", np.zeros(args.num_envs, dtype=bool))[i]:
                            won = int(infos["game_result"]["winner_index"][i]) == 0
                        else:
                            won = float(next_obs["opponent_scalars"][i][0]) <= 0
                        if won:
                            win_count += 1
                        recent_wins.append(1 if won else 0)

                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)

                        # Life totals from terminal obs (agent_scalars[0] = life)
                        agent_life = float(next_obs["agent_scalars"][i][0])
                        opp_life = float(next_obs["opponent_scalars"][i][0])
                        recent_agent_life.append(agent_life)
                        recent_opp_life.append(opp_life)
                        writer.add_scalar("charts/agent_life_at_end", agent_life, global_step)
                        writer.add_scalar("charts/opponent_life_at_end", opp_life, global_step)

                        # Turns per game from game_result (batched by _add_info)
                        if "game_result" in infos and infos.get("_game_result", np.zeros(args.num_envs, dtype=bool))[i]:
                            turns = int(infos["game_result"]["turns_played"][i])
                            recent_turns.append(turns)
                            writer.add_scalar("charts/turns_per_game", turns, global_step)

        # --- GAE ---
        with torch.no_grad():
            obs_t = obs_to_tensor(next_obs, device)
            next_value = agent.get_value(obs_t).flatten()

            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # --- Optimization phase ---
        # Flatten batch
        b_obs = {key: torch.tensor(obs_buf[key].reshape((-1,) + obs_buf[key].shape[2:]), device=device)
                 for key in OBS_KEYS}
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = {key: b_obs[key][mb_inds] for key in OBS_KEYS}
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs, b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # --- Logging ---
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        win_rate = win_count / max(episode_count, 1)

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/win_rate", win_rate, global_step)
        writer.add_scalar("charts/episodes", episode_count, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        # Smoothed game metrics (rolling window via deque maxlen)
        if recent_returns:
            writer.add_scalar("charts/avg_return_1000", np.mean(recent_returns), global_step)
        if recent_wins:
            writer.add_scalar("charts/win_rate_1000", np.mean(recent_wins), global_step)
        if recent_turns:
            writer.add_scalar("charts/avg_turns_per_game_1000", np.mean(recent_turns), global_step)
        if recent_agent_life:
            writer.add_scalar("charts/avg_agent_life_at_end_1000", np.mean(recent_agent_life), global_step)
        if recent_opp_life:
            writer.add_scalar("charts/avg_opponent_life_at_end_1000", np.mean(recent_opp_life), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # Decision type frequency (only log types that actually fire)
        total_decisions = sum(decision_type_counts.values()) or 1
        for dt_idx, name in enumerate(DECISION_TYPE_NAMES):
            count = decision_type_counts.get(dt_idx, 0)
            if count > 0:
                writer.add_scalar(f"decisions/count_{name}", count, global_step)
                writer.add_scalar(f"decisions/frac_{name}", count / total_decisions, global_step)

        # Mulligan quality metrics
        if mulligan_events:
            actions_taken = [a for _, a, _ in mulligan_events]
            mull_rate = np.mean(actions_taken)
            writer.add_scalar("mulligan/overall_mull_rate", mull_rate, global_step)
            writer.add_scalar("mulligan/num_decisions", len(mulligan_events), global_step)

            # Keep stats
            keeps = [l for l, a, _ in mulligan_events if a == 0]
            if keeps:
                writer.add_scalar("mulligan/avg_lands_when_keeping", np.mean(keeps), global_step)

            # Mulligan rate by land count bucket
            for lc in range(8):
                events = [a for l, a, _ in mulligan_events if l == lc]
                if events:
                    writer.add_scalar(f"mulligan/mull_rate_{lc}_lands", np.mean(events), global_step)

            # Key buckets: 0 lands and 6+ lands (should both be ~1.0)
            zero_land = [a for l, a, _ in mulligan_events if l == 0]
            if zero_land:
                writer.add_scalar("mulligan/mull_rate_0_lands_highlight", np.mean(zero_land), global_step)
            six_plus = [a for l, a, _ in mulligan_events if l >= 6]
            if six_plus:
                writer.add_scalar("mulligan/mull_rate_6plus_lands_highlight", np.mean(six_plus), global_step)

        decision_type_counts.clear()
        mulligan_events.clear()

        if update % 10 == 0 or update == 1:
            avg_return = np.mean(recent_returns) if recent_returns else 0.0
            print(
                f"Update {update}/{num_updates} | "
                f"Step {global_step:,} | "
                f"SPS {sps} | "
                f"Episodes {episode_count} | "
                f"Win rate {win_rate:.3f} | "
                f"Avg return {avg_return:.3f} | "
                f"PG loss {pg_loss.item():.4f} | "
                f"V loss {v_loss.item():.4f} | "
                f"Entropy {entropy_loss.item():.4f}"
            )

        # Periodic checkpoint every 50 updates
        if update % 50 == 0:
            ckpt_path = f"runs/{args.run_name}/agent.pt"
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "episode_count": episode_count,
                "win_count": win_count,
            }, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path} (update {update})")

    # Save final model
    model_path = f"runs/{args.run_name}/agent.pt"
    torch.save({
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "update": update,
        "episode_count": episode_count,
        "win_count": win_count,
    }, model_path)
    print(f"Model saved to {model_path}")

    envs.close()
    writer.close()
    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
