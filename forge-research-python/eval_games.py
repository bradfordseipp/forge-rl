"""Evaluation script: play N games with a trained agent and produce a play data report."""

import argparse
import atexit
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from forge_rl.env import ForgeRlEnv
from train_ppo import (
    Agent,
    DECISION_TYPE_NAMES,
    OBS_KEYS,
    start_servers,
    stop_servers,
)


# ---------------------------------------------------------------------------
# Action categorization
# ---------------------------------------------------------------------------

def get_decision_type_name(dt_index: int) -> str:
    if 0 <= dt_index < len(DECISION_TYPE_NAMES):
        return DECISION_TYPE_NAMES[dt_index]
    return f"unknown_{dt_index}"


def categorize_action(decision_type: str, description: str) -> tuple[str, str]:
    """Return (category, card_name) for a given action.

    Categories:
        Play Land, Cast Creature, Cast Spell, Activate, Attack, Block,
        Pass, Mulligan, Keep, Target, Other

    SpellAbility.toString() formats from Java:
        Creatures:   "Goblin Tomb Raider - Creature 1 / 2"
        Spells:      "Lightning Bolt deals 3 damage to any target."
        Lands:       "Play land" or just "Mountain" (land ability)
        Activated:   "{T}: Thermo-Alchemist deals 1 damage to each opponent."
        Targets:     "Player: AI_Opponent" or "Outlaw Medic (23)"
    """
    desc = description.strip()

    if decision_type == "mulligan":
        if desc.lower().startswith("keep") or desc == "0":
            return "Keep", ""
        return "Mulligan", ""

    if decision_type == "declare_attackers":
        # Extract creature name from "Attack with Goblin Tomb Raider (2/2)"
        match = re.match(r"Attack with (.+?)(?:\s*\(\d+/\d+\))?$", desc)
        card_name = match.group(1) if match else desc
        return "Attack", card_name

    if decision_type == "declare_blockers":
        match = re.match(r"(?:Block with |Done declaring blockers)(.+?)(?:\s*\(\d+/\d+\))?$", desc)
        card_name = match.group(1).strip() if match else desc
        return "Block", card_name

    if decision_type == "choose_entity":
        # Targeting decisions — "Player: AI_Opponent" or "Outlaw Medic (23)"
        return "Target", desc

    if decision_type == "choose_cards":
        return "Choose Cards", desc

    if decision_type == "choose_spell_ability":
        lower = desc.lower()

        # Pass / done
        if lower in ("pass", "done", "pass priority") or lower.startswith("pass"):
            return "Pass", ""

        # Land plays: "Play land", "Play land by ..."
        if lower.startswith("play land"):
            return "Play Land", ""

        # Explicit "Play <name>" or "Cast <name>"
        if desc.startswith("Play "):
            card_name = desc[5:].strip()
            return "Play Land", card_name
        if desc.startswith("Cast "):
            card_name = desc[5:].strip()
            card_name = re.sub(r"\s*\{[^}]*\}.*$", "", card_name)
            return f"Cast Spell", card_name

        # Activated abilities: start with "{T}:" or "{cost}:" patterns
        if re.match(r"^\{.*\}.*:", desc):
            # Extract ability source — often at end after "of" or in parentheses
            # e.g. "{T}: Thermo-Alchemist deals 1 damage to each opponent."
            # Try to find card name from the ability text
            match = re.search(r":\s*(\w[\w\s-]+?)(?:\s+deals|\s+gets|\s+adds|\s+creates|\s+puts|\s+exiles|\s+draws|\s+gains)", desc)
            card_name = match.group(1).strip() if match else ""
            return "Activate", card_name

        # Creature spells: "Name - Creature P / T"
        match = re.match(r"^(.+?)\s*-\s*Creature\s+\d+\s*/\s*\d+", desc)
        if match:
            card_name = match.group(1).strip()
            return "Cast Creature", card_name

        # Other spells: "Name deals/exiles/destroys..." or "Name - Enchantment/Artifact/..."
        # Sorceries/Instants typically show effect text: "Lightning Bolt deals 3 damage..."
        match = re.match(r"^(.+?)\s*-\s*(Enchantment|Artifact|Sorcery|Instant|Planeswalker)", desc)
        if match:
            card_name = match.group(1).strip()
            return "Cast Spell", card_name

        # Effect text like "Lightning Bolt deals 3 damage to any target."
        # or "Exile the top two cards of your library..."
        # These are spell casts — try to extract the card name (first few words before a verb)
        match = re.match(r"^([A-Z][\w\s'-]+?)\s+(?:deals|exiles|destroys|creates|puts|adds|target|returns|searches|draws|gains|prevents)", desc)
        if match:
            card_name = match.group(1).strip()
            # Remove trailing card IDs like "(38)"
            card_name = re.sub(r"\s*\(\d+\)\s*$", "", card_name)
            return "Cast Spell", card_name

        # Effect text starting with a verb/lowercase — sorcery/instant effects
        # e.g. "Exile the top two cards of your library..."
        if re.match(r"^[A-Z][a-z]", desc) and any(w in lower for w in
                     ("exile", "destroy", "return", "search", "draw", "discard",
                      "sacrifice", "choose", "look at", "reveal", "counter",
                      "each opponent", "each player", "you may")):
            return "Cast Spell", ""

        # Descriptions with "by CardName" suffix — impulse draw plays
        match = re.search(r"by\s+(.+?)(?:\s*\(\d+\))?$", desc)
        if match:
            card_name = match.group(1).strip()
            return "Cast Spell", card_name

        # Single word or card name (likely a land or simple ability)
        # Check if it's a common land name
        if desc in ("Mountain", "Great Furnace", "Island", "Plains", "Swamp", "Forest"):
            return "Play Land", desc

        # Remaining card names without type info — likely a spell cast
        # (cards like "Experimental Synthesizer", "Reckless Impulse")
        if re.match(r"^[A-Z][\w\s'-]+$", desc) and len(desc.split()) <= 4:
            return "Cast Spell", desc

        # Fallback
        return "Other", desc

    # All other decision types
    return "Other", desc


def broad_category(category: str) -> str:
    """Map detailed category to broad bucket for overall distribution."""
    if category in ("Cast Creature", "Cast Spell"):
        return category
    if category == "Activate":
        return "Activate"
    if category in ("Play Land", "Pass", "Attack", "Block", "Keep",
                     "Mulligan", "Target", "Choose Cards"):
        return category
    return "Other"


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def obs_to_tensor_single(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert single-env obs (no batch dim) to tensors with batch dim=1."""
    return {k: torch.tensor(obs[k], device=device).unsqueeze(0) for k in OBS_KEYS}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@dataclass
class ActionRecord:
    game_id: int
    turn: int
    decision_type: str
    action_index: int
    action_description: str
    category: str
    card_name: str


def run_evaluation(
    checkpoint: str,
    num_games: int,
    deck_a: str,
    deck_b: str,
    greedy: bool,
    port: int,
    server_dir: str,
) -> tuple[list[ActionRecord], list[dict]]:
    """Play num_games and return (action_records, game_results)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load agent
    agent = Agent().to(device)
    agent.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    agent.eval()
    print(f"Loaded agent from {checkpoint}")
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # Start server
    print(f"Starting game server on port {port}...")
    procs = start_servers(
        num_servers=1,
        base_port=port,
        server_dir=server_dir,
        deck_paths=[deck_a, deck_b],
    )
    atexit.register(stop_servers, procs)

    # Create env (no reward shaping)
    env = ForgeRlEnv(
        deck_path_a=deck_a,
        deck_path_b=deck_b,
        agent_player_index=0,
        port=port,
    )

    action_records: list[ActionRecord] = []
    game_results: list[dict] = []

    for game_id in range(num_games):
        obs, info = env.reset()
        terminated = False

        while not terminated:
            turn = int(obs["game_info"][0])
            dt_index = int(obs["decision_type"].argmax())
            dt_name = get_decision_type_name(dt_index)

            # Get action from agent
            with torch.no_grad():
                obs_t = obs_to_tensor_single(obs, device)
                action, _, _, _ = agent.get_action_and_value(obs_t)

                if greedy:
                    # Override: pick highest-probability legal action
                    trunk = agent._get_trunk(obs_t)
                    action_enc = agent._encode_actions(obs_t["action_features"])
                    trunk_proj = agent.trunk_to_action(trunk)
                    logits = torch.einsum("bd,bnd->bn", trunk_proj, action_enc)
                    mask = obs_t["action_mask"].bool()
                    logits[~mask] = -1e8
                    action = logits.argmax(dim=-1)

            action_idx = int(action.item())

            # Find description for chosen action
            action_desc = ""
            if "legal_actions" in info:
                for la in info["legal_actions"]:
                    if la["index"] == action_idx:
                        action_desc = la["description"]
                        break

            category, card_name = categorize_action(dt_name, action_desc)

            action_records.append(ActionRecord(
                game_id=game_id,
                turn=turn,
                decision_type=dt_name,
                action_index=action_idx,
                action_description=action_desc,
                category=category,
                card_name=card_name,
            ))

            obs, reward, terminated, truncated, info = env.step(action_idx)

        # Record game result
        result = {
            "game_id": game_id,
            "reward": reward,
            "win": reward > 0,
        }
        if "game_result" in info:
            result.update(info["game_result"])
        game_results.append(result)

        wins_so_far = sum(1 for g in game_results if g["win"])
        if (game_id + 1) % 10 == 0 or game_id == 0:
            print(f"  Game {game_id + 1}/{num_games} — "
                  f"Wins: {wins_so_far}/{game_id + 1} "
                  f"({100 * wins_so_far / (game_id + 1):.1f}%)")

    env.close()
    stop_servers(procs)
    return action_records, game_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    checkpoint: str,
    num_games: int,
    records: list[ActionRecord],
    results: list[dict],
) -> str:
    lines: list[str] = []

    def line(s: str = ""):
        lines.append(s)

    wins = sum(1 for g in results if g["win"])
    turns_list = [g.get("turns_played", 0) for g in results if g.get("turns_played")]
    avg_turns = np.mean(turns_list) if turns_list else 0.0

    line(f"Evaluation Report: {checkpoint}")
    line(f"Games: {num_games}, Wins: {wins} ({100 * wins / max(num_games, 1):.1f}%), "
         f"Avg turns: {avg_turns:.1f}")
    line()

    # === Mulligan Decisions ===
    mull_records = [r for r in records if r.decision_type == "mulligan"]
    if mull_records:
        keeps = sum(1 for r in mull_records if r.category == "Keep")
        mulls = sum(1 for r in mull_records if r.category == "Mulligan")
        total_mull = keeps + mulls
        line("=== Mulligan Decisions ===")
        if total_mull > 0:
            line(f"  Keep       {100 * keeps / total_mull:5.1f}%  ({keeps}/{total_mull})")
            line(f"  Mulligan   {100 * mulls / total_mull:5.1f}%  ({mulls}/{total_mull})")
        line()

    # === Per-turn breakdown ===
    # Group non-mulligan records by turn
    play_records = [r for r in records if r.decision_type != "mulligan"]

    # Determine which turns had data, cap at turn 10
    turns_seen: dict[int, list[ActionRecord]] = defaultdict(list)
    for r in play_records:
        t = min(r.turn, 10)  # aggregate 10+ together
        turns_seen[t].append(r)

    # Count how many games had actions on each turn
    games_per_turn: dict[int, int] = {}
    for t, recs in turns_seen.items():
        games_per_turn[t] = len(set(r.game_id for r in recs))

    for turn_num in sorted(turns_seen.keys()):
        recs = turns_seen[turn_num]
        n_games_this_turn = games_per_turn[turn_num]

        label = f"Turn {turn_num}" if turn_num < 10 else "Turn 10+"
        line(f"=== {label} ({n_games_this_turn} games) ===")

        # Count categories
        cat_counter: Counter = Counter()
        cat_cards: dict[str, Counter] = defaultdict(Counter)
        for r in recs:
            bc = broad_category(r.category)
            cat_counter[bc] += 1
            if r.card_name:
                cat_cards[bc][r.card_name] += 1
            # For Cast, also track specific card in detailed category
            if r.category.startswith("Cast:") or r.category.startswith("Activate:"):
                cat_counter[r.category] = cat_counter.get(r.category, 0)  # ensure exists
                # We already counted in broad; just track card names under broad

        total_actions = sum(cat_counter.values())
        for cat, count in cat_counter.most_common():
            pct = 100 * count / total_actions
            # Build card breakdown
            card_info = ""
            if cat in cat_cards and cat_cards[cat]:
                top_cards = cat_cards[cat].most_common(5)
                card_total = sum(c for _, c in top_cards)
                parts = []
                for cname, ccount in top_cards:
                    parts.append(f"{cname} {100 * ccount / sum(cat_cards[cat].values()):.0f}%")
                card_info = f"  [{', '.join(parts)}]"

            # Special annotation for Attack: average creatures
            extra = ""
            if cat == "Attack":
                attack_recs = [r for r in recs if broad_category(r.category) == "Attack"]
                # Each attack record is one attack decision (may describe multiple creatures)
                extra = f"  ({count} decisions)"

            line(f"  {cat:<25s} {pct:5.1f}%{card_info}{extra}")

        line()

    # === Overall Action Distribution ===
    line("=== Overall Action Distribution ===")
    overall = Counter()
    overall_cards: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        bc = broad_category(r.category)
        overall[bc] += 1
        if r.card_name:
            overall_cards[bc][r.card_name] += 1

    total = sum(overall.values())
    for cat, count in overall.most_common():
        pct = 100 * count / total
        card_info = ""
        if cat in overall_cards and overall_cards[cat]:
            top_cards = overall_cards[cat].most_common(5)
            parts = []
            for cname, ccount in top_cards:
                parts.append(f"{cname} {100 * ccount / sum(overall_cards[cat].values()):.0f}%")
            card_info = f"  [{', '.join(parts)}]"
        line(f"  {cat:<25s} {pct:5.1f}%{card_info}")

    line()

    # === Top Cards Played/Cast ===
    line("=== Top Cards (Cast + Played) ===")
    card_counts: Counter = Counter()
    for r in records:
        if r.card_name and r.category in ("Cast Creature", "Cast Spell", "Play Land"):
            card_counts[r.card_name] += 1
    for cname, ccount in card_counts.most_common(15):
        line(f"  {cname:<30s} {ccount}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Forge RL agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to agent .pt checkpoint")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games to play (default: 100)")
    parser.add_argument("--output", type=str, default="eval_report.txt",
                        help="Output file path (default: eval_report.txt)")
    parser.add_argument("--deck-a", type=str,
                        default="src/main/resources/decks/mono_red_pingers.dck",
                        help="Deck A path (agent's deck)")
    parser.add_argument("--deck-b", type=str,
                        default="src/main/resources/decks/caw_gates.dck",
                        help="Deck B path (opponent's deck)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy (argmax) action selection instead of sampling")
    parser.add_argument("--port", type=int, default=50061,
                        help="gRPC server port (default: 50061, avoids training ports)")
    parser.add_argument("--server-jar-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "forge-research"),
                        help="Path to forge-research directory")
    args = parser.parse_args()

    print(f"Evaluating {args.checkpoint}")
    print(f"  Deck A: {args.deck_a}")
    print(f"  Deck B: {args.deck_b}")
    print(f"  Games: {args.num_games}")
    print(f"  Greedy: {args.greedy}")
    print()

    t0 = time.time()
    records, results = run_evaluation(
        checkpoint=args.checkpoint,
        num_games=args.num_games,
        deck_a=args.deck_a,
        deck_b=args.deck_b,
        greedy=args.greedy,
        port=args.port,
        server_dir=args.server_jar_dir,
    )
    elapsed = time.time() - t0

    report = generate_report(args.checkpoint, args.num_games, records, results)
    report += f"\n\nCompleted in {elapsed:.1f}s ({elapsed / max(args.num_games, 1):.2f}s/game)\n"

    # Print to stdout
    print()
    print(report)

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
