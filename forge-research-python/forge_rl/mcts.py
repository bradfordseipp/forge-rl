"""MCTS inference layer using a trained policy/value network.

Wraps a PPO-trained Agent to guide Monte Carlo Tree Search at inference time.
Uses the Forge gRPC Simulate RPC to run forward simulations on cloned games.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from forge_rl.client import ForgeRlClient
from forge_rl.env import (
    CARD_FEATURES, MAX_ACTIONS, MAX_BATTLEFIELD, MAX_EXILE,
    MAX_GRAVEYARD, MAX_HAND, MAX_STACK, NUM_DECISION_TYPES,
    STACK_FEATURES, ACTION_FEATURES,
)


# Decision types worth running MCTS on (high-impact strategic decisions)
MCTS_DECISION_TYPES = {0, 1}  # CHOOSE_SPELL_ABILITY, DECLARE_ATTACKERS


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    action_index: int = -1          # action that led to this node (-1 for root)
    parent: Optional[MCTSNode] = None
    children: dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0            # N
    total_value: float = 0.0        # W
    prior: float = 0.0              # P (from policy network)
    is_terminal: bool = False
    terminal_reward: float = 0.0

    @property
    def q_value(self) -> float:
        """Mean value Q = W / N."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        """PUCT selection score: Q + c * P * sqrt(N_parent) / (1 + N)."""
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration


class MCTSAgent:
    """MCTS agent that uses a trained policy/value network for guided search.

    At each decision point, runs N simulations via the Forge gRPC Simulate RPC,
    builds a search tree using PUCT, and returns the most-visited action.
    """

    def __init__(
        self,
        agent: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        num_simulations: int = 30,
        c_puct: float = 1.5,
        max_sim_depth: int = 3,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.agent = agent
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_sim_depth = max_sim_depth
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.agent.eval()

    def select_action(
        self,
        client: ForgeRlClient,
        obs: dict[str, np.ndarray],
        decision_type: int,
    ) -> int:
        """Select an action using MCTS or direct policy.

        For high-impact decisions (spells, attackers), runs full MCTS.
        For other decisions, uses the policy network directly.

        Args:
            client: gRPC client connected to the game server
            obs: current observation dict (numpy arrays)
            decision_type: the DecisionType enum value

        Returns:
            action_index to take
        """
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask > 0)[0]

        if len(legal_actions) <= 1:
            return legal_actions[0] if len(legal_actions) == 1 else 0

        # Only run MCTS for high-impact decisions
        if decision_type not in MCTS_DECISION_TYPES:
            return self._policy_action(obs)

        return self._mcts_action(client, obs, legal_actions)

    def _policy_action(self, obs: dict[str, np.ndarray]) -> int:
        """Select action using policy network directly (greedy)."""
        obs_t = {k: torch.tensor(obs[k], device=self.device).unsqueeze(0) for k in obs}
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(obs_t)
        return action.item()

    def _get_policy_and_value(self, obs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        """Get policy probabilities and value estimate from the network."""
        obs_t = {k: torch.tensor(obs[k], device=self.device).unsqueeze(0) for k in obs}
        with torch.no_grad():
            action, _, _, value = self.agent.get_action_and_value(obs_t)

            # Get raw logits for policy priors
            # Re-run trunk and action encoding to get logits
            scalar_enc, card_tokens, card_mask, stack_enc = self.agent._encode_board(obs_t)
            action_tokens = self.agent._encode_actions(obs_t["action_features"])
            safe_mask = card_mask.clone()
            all_masked = card_mask.all(dim=1)
            if all_masked.any():
                safe_mask[all_masked, 0] = False
            attn_out, _ = self.agent.action_card_cross_attn(
                query=action_tokens, key=card_tokens, value=card_tokens,
                key_padding_mask=safe_mask,
            )
            action_repr = self.agent.action_cross_attn_norm(action_tokens + attn_out)
            context = torch.cat([scalar_enc, stack_enc], dim=-1)
            context_expanded = context.unsqueeze(1).expand(-1, 256, -1)
            score_input = torch.cat([action_repr, context_expanded], dim=-1)
            logits = self.agent.action_score(score_input).squeeze(-1)  # (1, 256)

            # Apply mask and softmax to get priors
            mask = obs_t["action_mask"]
            masked_logits = logits.clone()
            masked_logits[mask == 0] = -1e8
            priors = torch.softmax(masked_logits, dim=-1).squeeze(0).cpu().numpy()

        return priors, value.item()

    def _mcts_action(
        self,
        client: ForgeRlClient,
        obs: dict[str, np.ndarray],
        legal_actions: np.ndarray,
    ) -> int:
        """Run MCTS and return the best action."""
        # Get policy priors and value for root
        priors, root_value = self._get_policy_and_value(obs)

        # Create root node
        root = MCTSNode()
        root.visit_count = 1
        root.total_value = root_value

        # Initialize children with policy priors
        for action_idx in legal_actions:
            child = MCTSNode(
                action_index=action_idx,
                parent=root,
                prior=priors[action_idx],
            )
            root.children[action_idx] = child

        # Add Dirichlet noise to root priors for exploration
        if self.dirichlet_epsilon > 0 and len(legal_actions) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for i, action_idx in enumerate(legal_actions):
                child = root.children[action_idx]
                child.prior = (
                    (1 - self.dirichlet_epsilon) * child.prior
                    + self.dirichlet_epsilon * noise[i]
                )

        # Run simulations
        for _ in range(self.num_simulations):
            # Selection: pick child with highest PUCT score
            best_action = max(
                root.children.keys(),
                key=lambda a: root.children[a].puct_score(self.c_puct, root.visit_count),
            )
            child = root.children[best_action]

            if child.is_terminal:
                # Terminal node — use cached reward
                value = child.terminal_reward
            else:
                # Simulate: clone game, take action, run forward
                try:
                    sim_result = client.simulate(
                        first_action_index=best_action,
                        max_agent_decisions=self.max_sim_depth,
                    )

                    if sim_result.game_ended:
                        child.is_terminal = True
                        child.terminal_reward = sim_result.terminal_reward
                        value = sim_result.terminal_reward
                    else:
                        # Evaluate leaf with value network
                        leaf_obs = self._proto_to_obs(sim_result.leaf_observation)
                        _, value = self._get_policy_and_value(leaf_obs)
                except Exception as e:
                    # Simulation failed — use root value as fallback
                    value = root_value

            # Backpropagate
            child.visit_count += 1
            child.total_value += value
            root.visit_count += 1

        # Select action with most visits
        best_action = max(root.children.keys(), key=lambda a: root.children[a].visit_count)

        return best_action

    def _proto_to_obs(self, obs_proto) -> dict[str, np.ndarray]:
        """Convert a protobuf Observation to the numpy dict format expected by the agent.

        This is a simplified version — only builds enough for value network evaluation.
        """
        from forge_rl.env import ForgeRlEnv
        # Reuse env's observation building logic with a dummy decision point
        result = {}

        gi = obs_proto.game_info
        result["game_info"] = np.array([
            gi.turn, gi.phase, gi.active_player_index, gi.priority_player_index,
            gi.agent_mulligan_count, gi.opponent_mulligan_count,
        ], dtype=np.int32)

        result["agent_scalars"] = self._player_scalars(obs_proto.agent_player)
        result["opponent_scalars"] = self._player_scalars(obs_proto.opponent_player)

        result["agent_hand"] = self._card_matrix(obs_proto.agent_player.hand, MAX_HAND)
        result["agent_battlefield"] = self._card_matrix(obs_proto.agent_player.battlefield, MAX_BATTLEFIELD)
        result["opponent_battlefield"] = self._card_matrix(obs_proto.opponent_player.battlefield, MAX_BATTLEFIELD)
        result["agent_graveyard"] = self._card_matrix(obs_proto.agent_player.graveyard, MAX_GRAVEYARD)
        result["opponent_graveyard"] = self._card_matrix(obs_proto.opponent_player.graveyard, MAX_GRAVEYARD)
        result["agent_exile"] = self._card_matrix(obs_proto.agent_player.exile, MAX_EXILE)
        result["opponent_exile"] = self._card_matrix(obs_proto.opponent_player.exile, MAX_EXILE)

        result["stack"] = self._stack_matrix(obs_proto.stack)
        result["action_mask"] = np.zeros(MAX_ACTIONS, dtype=np.int8)
        result["decision_type"] = np.zeros(NUM_DECISION_TYPES, dtype=np.int8)
        result["action_features"] = np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.float32)

        return result

    @staticmethod
    def _player_scalars(player_state) -> np.ndarray:
        mp = player_state.mana_pool
        return np.array([
            player_state.life, player_state.hand_size, player_state.library_size,
            player_state.lands_played, player_state.max_lands,
            mp.white, mp.blue, mp.black, mp.red, mp.green, mp.colorless,
        ], dtype=np.int32)

    @staticmethod
    def _card_matrix(cards, max_cards: int) -> np.ndarray:
        mat = np.full((max_cards, CARD_FEATURES), -1, dtype=np.float32)
        for i, card in enumerate(cards):
            if i >= max_cards:
                break
            colors = card.colors_bitmask
            types = card.type_bitmask
            kw = card.keyword_bitmask
            mat[i] = [
                card.name_id,
                card.power / 20.0, card.toughness / 20.0, card.cmc / 10.0,
                int(card.tapped), int(card.summoning_sick),
                (colors >> 0) & 1, (colors >> 1) & 1, (colors >> 2) & 1,
                (colors >> 3) & 1, (colors >> 4) & 1,
                card.damage / 20.0, card.loyalty / 7.0,
                int(card.attacking), int(card.blocking),
                card.counter_count / 10.0,
                (types >> 0) & 1, (types >> 1) & 1, (types >> 2) & 1,
                (types >> 3) & 1, (types >> 4) & 1, (types >> 5) & 1,
                (types >> 6) & 1,
                (kw >> 0) & 1, (kw >> 1) & 1, (kw >> 2) & 1, (kw >> 3) & 1,
                (kw >> 4) & 1, (kw >> 5) & 1, (kw >> 6) & 1, (kw >> 7) & 1,
                (kw >> 8) & 1, (kw >> 9) & 1, (kw >> 10) & 1, (kw >> 11) & 1,
                (kw >> 12) & 1, (kw >> 13) & 1,
                card.plus_one_counter_count / 10.0,
            ]
        return mat

    @staticmethod
    def _stack_matrix(stack_entries) -> np.ndarray:
        mat = np.full((MAX_STACK, STACK_FEATURES), -1, dtype=np.float32)
        for i, entry in enumerate(stack_entries):
            if i >= MAX_STACK:
                break
            mat[i] = [entry.source_name_id, entry.controller_index]
        return mat
