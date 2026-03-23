"""Evaluate MCTS agent vs baselines with detailed metrics.

Plays N games comparing policy-only vs MCTS at various simulation budgets.
Reports win rate, action agreement, per-decision-type stats, and latency.

Usage:
    python eval_mcts.py --checkpoint runs/run_crossattn_v1/agent.pt \
        --onnx-model runs/run_crossattn_v1/agent.onnx \
        --num-games 100 --sim-budgets 0,10,30

    sim-budget 0 = policy-only (no MCTS)
"""

import argparse
import atexit
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from forge_rl.client import ForgeRlClient
from forge_rl.env import ForgeRlEnv, ACTION_FEATURES, CARD_FEATURES, MAX_ACTIONS, STACK_FEATURES
from forge_rl.mcts import MCTSAgent, MCTS_DECISION_TYPES
from train_ppo import Agent, OBS_KEYS


DECISION_TYPE_NAMES = [
    "choose_spell_ability", "declare_attackers", "declare_blockers",
    "choose_targets", "mulligan", "choose_cards", "confirm_action",
    "choose_entity", "choose_color", "choose_number", "choose_type",
    "choose_pile", "choose_mode", "order_cards", "ai_fallback",
]


@dataclass
class GameStats:
    won: bool = False
    turns: int = 0
    agent_life: int = 0
    opp_life: int = 0
    total_decisions: int = 0
    mcts_decisions: int = 0
    mcts_overrides: int = 0  # times MCTS picked a different action than policy
    decision_type_counts: dict = field(default_factory=lambda: defaultdict(int))
    decision_type_mcts_overrides: dict = field(default_factory=lambda: defaultdict(int))
    decision_latencies: list = field(default_factory=list)
    game_time: float = 0.0


def play_game(
    client: ForgeRlClient,
    mcts_agent: MCTSAgent | None,
    deck_a: str,
    deck_b: str,
    onnx_path: str,
    seed: int,
) -> GameStats:
    """Play a single game and collect stats."""
    stats = GameStats()
    t_game_start = time.time()

    resp = client.reset(
        deck_path_a=deck_a,
        deck_path_b=deck_b,
        agent_player_index=0,
        seed=seed,
        dual_rl=True,
        opponent_model_path=onnx_path,
    )

    terminated = False
    obs_proto = resp.observation
    dp = resp.decision_point

    while not terminated:
        if dp is None or len(dp.legal_actions) == 0:
            break

        dt = dp.type
        stats.total_decisions += 1
        stats.decision_type_counts[dt] += 1
        num_legal = len(dp.legal_actions)

        if num_legal == 1:
            action = dp.legal_actions[0].index
        elif mcts_agent is not None:
            # Build obs dict from proto for the agent
            obs = _proto_to_obs(obs_proto, dp)

            # Get policy action for comparison
            policy_action = mcts_agent._policy_action(obs)

            # Get MCTS action (or policy-only if not an MCTS decision type)
            t_start = time.time()
            mcts_action = mcts_agent.select_action(client, obs, dt)
            t_elapsed = time.time() - t_start

            action = mcts_action

            if dt in MCTS_DECISION_TYPES and num_legal > 1:
                stats.mcts_decisions += 1
                stats.decision_latencies.append(t_elapsed)
                if mcts_action != policy_action:
                    stats.mcts_overrides += 1
                    stats.decision_type_mcts_overrides[dt] += 1
        else:
            # Policy-only: use first legal action as fallback (shouldn't happen)
            action = dp.legal_actions[0].index

        resp = client.step(action_index=action)
        terminated = resp.terminated
        obs_proto = resp.observation
        dp = resp.decision_point if not terminated else None

        if resp.HasField("game_result"):
            gr = resp.game_result
            stats.won = gr.winner_index == 0
            stats.turns = gr.turns_played

    stats.agent_life = obs_proto.agent_player.life
    stats.opp_life = obs_proto.opponent_player.life
    stats.game_time = time.time() - t_game_start

    return stats


def _proto_to_obs(obs_proto, decision_point) -> dict[str, np.ndarray]:
    """Convert protobuf observation + decision point to numpy dict.

    Reuses ForgeRlEnv's logic by creating a temporary instance method call.
    """
    # Build a lightweight env-like converter
    from forge_rl.env import (
        MAX_HAND, MAX_BATTLEFIELD, MAX_GRAVEYARD, MAX_EXILE, MAX_STACK,
        NUM_DECISION_TYPES,
    )

    result = {}

    gi = obs_proto.game_info
    result["game_info"] = np.array([
        gi.turn, gi.phase, gi.active_player_index, gi.priority_player_index,
        gi.agent_mulligan_count, gi.opponent_mulligan_count,
    ], dtype=np.int32)

    result["agent_scalars"] = _player_scalars(obs_proto.agent_player)
    result["opponent_scalars"] = _player_scalars(obs_proto.opponent_player)

    result["agent_hand"] = _card_matrix(obs_proto.agent_player.hand, MAX_HAND)
    result["agent_battlefield"] = _card_matrix(obs_proto.agent_player.battlefield, MAX_BATTLEFIELD)
    result["opponent_battlefield"] = _card_matrix(obs_proto.opponent_player.battlefield, MAX_BATTLEFIELD)
    result["agent_graveyard"] = _card_matrix(obs_proto.agent_player.graveyard, MAX_GRAVEYARD)
    result["opponent_graveyard"] = _card_matrix(obs_proto.opponent_player.graveyard, MAX_GRAVEYARD)
    result["agent_exile"] = _card_matrix(obs_proto.agent_player.exile, MAX_EXILE)
    result["opponent_exile"] = _card_matrix(obs_proto.opponent_player.exile, MAX_EXILE)

    result["stack"] = _stack_matrix(obs_proto.stack)

    # Action mask
    action_mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    if decision_point and decision_point.legal_actions:
        for action in decision_point.legal_actions:
            if 0 <= action.index < MAX_ACTIONS:
                action_mask[action.index] = 1
    result["action_mask"] = action_mask

    # Decision type one-hot
    decision_type = np.zeros(NUM_DECISION_TYPES, dtype=np.int8)
    if decision_point:
        dt = decision_point.type
        if 0 <= dt < NUM_DECISION_TYPES:
            decision_type[dt] = 1
    result["decision_type"] = decision_type

    # Action features
    action_features = np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.float32)
    if decision_point and decision_point.legal_actions:
        for action in decision_point.legal_actions:
            idx = action.index
            if 0 <= idx < MAX_ACTIONS:
                src_name_id = action.source_name_id
                src_card_id = action.source_card_id
                is_pass = 1.0 if (src_name_id == 0 and src_card_id == 0) else 0.0
                type_bm = action.source_type_bitmask
                kw_bm = action.source_keyword_bitmask
                action_features[idx] = [
                    float(src_name_id), np.log1p(src_card_id), is_pass,
                    1.0 if action.target_is_player else 0.0,
                    float(action.target_name_id), np.log1p(action.target_card_id),
                    1.0 if action.target_is_own else 0.0,
                    action.source_power / 20.0, action.source_toughness / 20.0,
                    action.source_cmc / 10.0,
                    1.0 if action.cost_taps_self else 0.0,
                    1.0 if action.cost_sacrifices else 0.0,
                    1.0 if action.cost_pays_life else 0.0,
                    1.0 if action.cost_exiles_from_graveyard else 0.0,
                    1.0 if action.can_target_creatures else 0.0,
                    1.0 if action.can_target_players else 0.0,
                    action.damage_amount / 20.0,
                    1.0 if (type_bm & 1) else 0.0,
                    1.0 if (type_bm & 2) else 0.0,
                    1.0 if (type_bm & 0xC) else 0.0,
                    (kw_bm >> 0) & 1, (kw_bm >> 1) & 1, (kw_bm >> 2) & 1,
                    (kw_bm >> 3) & 1, (kw_bm >> 4) & 1, (kw_bm >> 5) & 1,
                    (kw_bm >> 6) & 1, (kw_bm >> 7) & 1, (kw_bm >> 8) & 1,
                    (kw_bm >> 9) & 1, (kw_bm >> 10) & 1, (kw_bm >> 11) & 1,
                    (kw_bm >> 12) & 1, (kw_bm >> 13) & 1,
                ]
    result["action_features"] = action_features

    return result


def _player_scalars(player_state) -> np.ndarray:
    mp = player_state.mana_pool
    return np.array([
        player_state.life, player_state.hand_size, player_state.library_size,
        player_state.lands_played, player_state.max_lands,
        mp.white, mp.blue, mp.black, mp.red, mp.green, mp.colorless,
    ], dtype=np.int32)


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


def _stack_matrix(stack_entries) -> np.ndarray:
    mat = np.full((10, STACK_FEATURES), -1, dtype=np.float32)
    for i, entry in enumerate(stack_entries):
        if i >= 10:
            break
        mat[i] = [
            entry.source_name_id, entry.controller_index,
            entry.target_name_id,
            1.0 if entry.target_is_player else 0.0,
            1.0 if entry.target_is_own else 0.0,
            entry.source_type_bitmask & 1,
        ]
    return mat


def start_server(port: int, server_dir: str, deck_paths: list[str], log_dir: str | None = None):
    """Start a single Forge gRPC server."""
    cp_file = os.path.join(server_dir, "target", "classpath.txt")
    with open(cp_file) as f:
        deps_cp = f.read().strip()

    classpath = os.path.join(server_dir, "target", "classes")
    forge_root = os.path.dirname(os.path.abspath(server_dir))
    for mod in ["forge-core", "forge-game", "forge-ai", "forge-gui", "forge-gui-desktop"]:
        cdir = os.path.join(forge_root, mod, "target", "classes")
        if os.path.isdir(cdir):
            classpath += ":" + cdir
    classpath += ":" + deps_cp

    abs_decks = [str(Path(server_dir, dp).resolve()) for dp in deck_paths]
    cmd = ["java", "-Xmx256m", "-cp", classpath, "forge.research.ForgeResearchServer", str(port)] + abs_decks

    stdout_dest = subprocess.DEVNULL
    stderr_dest = subprocess.DEVNULL
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = open(Path(log_dir) / f"server_{port}.log", "w")
        stdout_dest = log_file
        stderr_dest = log_file

    proc = subprocess.Popen(cmd, cwd=server_dir, stdout=stdout_dest, stderr=stderr_dest)

    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with code {proc.returncode}")
        try:
            with socket.create_connection(("localhost", port), timeout=0.5):
                return proc
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"Server on port {port} did not start")


def run_eval(args):
    server_dir = str(Path(__file__).resolve().parent.parent / "forge-research")
    deck_a = args.deck_a
    deck_b = args.deck_b

    # Load model
    device = torch.device("cpu")  # MCTS eval on CPU for simplicity
    agent = Agent().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        agent.load_state_dict(ckpt["model_state_dict"])
    else:
        agent.load_state_dict(ckpt)
    agent.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    onnx_path = str(Path(args.onnx_model).resolve())
    print(f"ONNX model: {onnx_path}")

    # Parse simulation budgets
    sim_budgets = [int(x) for x in args.sim_budgets.split(",")]
    print(f"Simulation budgets: {sim_budgets}")
    print(f"Games per budget: {args.num_games}")

    # Start server
    port = args.port
    print(f"\nStarting server on port {port}...")
    proc = start_server(port, server_dir, [deck_a, deck_b])
    atexit.register(lambda: (proc.terminate(), proc.wait(timeout=5)))
    print("Server ready")

    client = ForgeRlClient(port=port)

    all_results = {}

    for budget in sim_budgets:
        label = f"policy_only" if budget == 0 else f"mcts_{budget}"
        print(f"\n{'='*60}")
        print(f"Evaluating: {label} ({args.num_games} games)")
        print(f"{'='*60}")

        if budget == 0:
            mcts_agent = MCTSAgent(agent, device=device, num_simulations=1, c_puct=0)
        else:
            mcts_agent = MCTSAgent(
                agent, device=device,
                num_simulations=budget,
                c_puct=args.c_puct,
                max_sim_depth=args.sim_depth,
            )

        game_stats = []
        for g in range(args.num_games):
            seed = args.seed + g
            try:
                stats = play_game(client, mcts_agent if budget > 0 else mcts_agent, deck_a, deck_b, onnx_path, seed)
                game_stats.append(stats)
                win_so_far = sum(1 for s in game_stats if s.won) / len(game_stats)
                if (g + 1) % 10 == 0 or g == 0:
                    print(f"  Game {g+1}/{args.num_games}: "
                          f"win_rate={win_so_far:.3f} "
                          f"({'W' if stats.won else 'L'} in {stats.turns}t, {stats.game_time:.1f}s)")
            except Exception as e:
                print(f"  Game {g+1} FAILED: {e}")
                # Reconnect
                client = ForgeRlClient(port=port)

        # Aggregate results
        if not game_stats:
            continue

        wins = sum(1 for s in game_stats if s.won)
        win_rate = wins / len(game_stats)
        avg_turns = np.mean([s.turns for s in game_stats])
        avg_game_time = np.mean([s.game_time for s in game_stats])

        total_mcts_decisions = sum(s.mcts_decisions for s in game_stats)
        total_mcts_overrides = sum(s.mcts_overrides for s in game_stats)
        agreement_rate = 1.0 - (total_mcts_overrides / max(total_mcts_decisions, 1))

        all_latencies = []
        for s in game_stats:
            all_latencies.extend(s.decision_latencies)

        # Per-decision-type breakdown
        dt_counts = defaultdict(int)
        dt_overrides = defaultdict(int)
        for s in game_stats:
            for dt, cnt in s.decision_type_counts.items():
                dt_counts[dt] += cnt
            for dt, cnt in s.decision_type_mcts_overrides.items():
                dt_overrides[dt] += cnt

        result = {
            "label": label,
            "sim_budget": budget,
            "num_games": len(game_stats),
            "win_rate": win_rate,
            "wins": wins,
            "avg_turns": float(avg_turns),
            "avg_game_time_sec": float(avg_game_time),
            "mcts_decisions_total": total_mcts_decisions,
            "mcts_overrides_total": total_mcts_overrides,
            "action_agreement_rate": agreement_rate,
            "latency_mean_ms": float(np.mean(all_latencies) * 1000) if all_latencies else 0,
            "latency_p50_ms": float(np.median(all_latencies) * 1000) if all_latencies else 0,
            "latency_p95_ms": float(np.percentile(all_latencies, 95) * 1000) if all_latencies else 0,
        }

        # Decision type breakdown
        dt_breakdown = {}
        for dt in sorted(dt_counts.keys()):
            name = DECISION_TYPE_NAMES[dt] if dt < len(DECISION_TYPE_NAMES) else f"type_{dt}"
            dt_breakdown[name] = {
                "count": dt_counts[dt],
                "mcts_overrides": dt_overrides.get(dt, 0),
                "override_rate": dt_overrides.get(dt, 0) / max(dt_counts[dt], 1),
            }
        result["decision_types"] = dt_breakdown

        all_results[label] = result

        # Print summary
        print(f"\n  Results for {label}:")
        print(f"    Win rate:           {win_rate:.1%} ({wins}/{len(game_stats)})")
        print(f"    Avg turns/game:     {avg_turns:.1f}")
        print(f"    Avg game time:      {avg_game_time:.1f}s")
        if budget > 0:
            print(f"    Action agreement:   {agreement_rate:.1%}")
            print(f"    MCTS overrides:     {total_mcts_overrides}/{total_mcts_decisions}")
            if all_latencies:
                print(f"    Latency (MCTS):     mean={result['latency_mean_ms']:.0f}ms "
                      f"p50={result['latency_p50_ms']:.0f}ms p95={result['latency_p95_ms']:.0f}ms")
            print(f"    Per-decision-type overrides:")
            for name, info in dt_breakdown.items():
                if info["count"] > 0 and name in ["choose_spell_ability", "declare_attackers"]:
                    print(f"      {name}: {info['mcts_overrides']}/{info['count']} "
                          f"({info['override_rate']:.1%} override rate)")

    # Comparison summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Budget':<15} {'Win Rate':<12} {'Agreement':<12} {'Latency(ms)':<12} {'Time/Game':<12}")
        print("-" * 63)
        for label, r in all_results.items():
            print(f"{r['label']:<15} {r['win_rate']:<12.1%} "
                  f"{r['action_agreement_rate']:<12.1%} "
                  f"{r['latency_mean_ms']:<12.0f} "
                  f"{r['avg_game_time_sec']:<12.1f}s")

    # Save JSON report
    report_path = args.output or f"eval_mcts_{int(time.time())}.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

    proc.terminate()


def main():
    parser = argparse.ArgumentParser(description="Evaluate MCTS agent")
    parser.add_argument("--checkpoint", required=True, help="Path to agent.pt checkpoint")
    parser.add_argument("--onnx-model", required=True, help="Path to agent.onnx for simulation")
    parser.add_argument("--num-games", type=int, default=50, help="Games per simulation budget")
    parser.add_argument("--sim-budgets", type=str, default="0,10,30",
                        help="Comma-separated simulation budgets (0=policy only)")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant")
    parser.add_argument("--sim-depth", type=int, default=3, help="Max agent decisions per simulation")
    parser.add_argument("--deck-a", type=str, default="src/main/resources/decks/mono_red_pingers.dck")
    parser.add_argument("--deck-b", type=str, default="src/main/resources/decks/caw_gates.dck")
    parser.add_argument("--port", type=int, default=50099, help="Server port (avoid training ports)")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed for games")
    parser.add_argument("--output", type=str, default="", help="Output JSON path")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
