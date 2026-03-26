"""Text-only game: play MTG using only the features the RL agent sees.

Shows the exact observation the model receives at each decision point,
decoded into human-readable format. You see NOTHING the model doesn't see.

Usage:
    python play_text.py [--deck-a DECK] [--deck-b DECK] [--port PORT]

Starts a Java server automatically.
"""

import atexit
import os
import sys
from pathlib import Path

import numpy as np

from forge_rl.client import ForgeRlClient
from forge_rl.env import (
    CARD_FEATURES, STACK_FEATURES, MAX_ACTIONS, MAX_HAND, MAX_BATTLEFIELD,
    MAX_GRAVEYARD, MAX_EXILE, MAX_STACK, NUM_DECISION_TYPES, ACTION_FEATURES,
)
from train_ppo import (
    card_name_to_id, parse_deck_card_names, start_servers, stop_servers,
    DECISION_TYPE_NAMES,
)

# Build reverse mapping: name_id → card name (from both decks)
KNOWN_CARDS = {}

KEYWORD_NAMES = [
    "flying", "first_strike", "double_strike", "deathtouch", "lifelink",
    "trample", "haste", "reach", "vigilance", "menace",
    "defender", "hexproof", "indestructible", "flash",
]

TYPE_NAMES = ["creature", "land", "instant", "sorcery", "enchantment", "artifact", "planeswalker"]
COLOR_NAMES = ["W", "U", "B", "R", "G"]


def register_deck(deck_path: str):
    """Add all cards from a deck to the known cards mapping."""
    # Resolve path
    if not os.path.isfile(deck_path):
        forge_dir = str(Path(__file__).resolve().parent.parent / "forge-research")
        resolved = os.path.join(forge_dir, deck_path)
        if os.path.isfile(resolved):
            deck_path = resolved
    with open(deck_path) as f:
        in_main = False
        for line in f:
            line = line.strip()
            if line == "[Main]":
                in_main = True
                continue
            if line.startswith("["):
                in_main = False
                continue
            if not in_main or not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                name = parts[1]
                nid = card_name_to_id(name)
                KNOWN_CARDS[nid] = name


def card_name(name_id: int) -> str:
    """Look up card name from name_id."""
    if name_id <= 0:
        return "—"
    return KNOWN_CARDS.get(name_id, f"unknown#{name_id}")


def format_card(features: np.ndarray) -> str:
    """Format a single card's features into a readable string."""
    nid = int(features[0])
    if nid <= 0:
        return None  # empty slot

    name = card_name(nid)
    power = features[1] * 20
    toughness = features[2] * 20
    cmc = features[3] * 10
    tapped = features[4] > 0.5
    sick = features[5] > 0.5
    damage = features[11] * 20
    counters = features[15] * 10
    p1p1 = features[37] * 10
    attacking = features[13] > 0.5
    blocking = features[14] > 0.5
    playable_exile = features[38] > 0.5 if len(features) > 38 else False

    # Types
    types = [TYPE_NAMES[i] for i in range(7) if features[16 + i] > 0.5]
    # Keywords
    kws = [KEYWORD_NAMES[i] for i in range(14) if features[23 + i] > 0.5]
    # Colors
    colors = "".join(COLOR_NAMES[i] for i in range(5) if features[6 + i] > 0.5)

    parts = [f"{name}"]
    if "creature" in types:
        parts.append(f"{power:.0f}/{toughness:.0f}")
    if cmc > 0:
        parts.append(f"cmc={cmc:.0f}")
    if colors:
        parts.append(f"[{colors}]")
    if tapped:
        parts.append("TAPPED")
    if sick:
        parts.append("SICK")
    if damage > 0:
        parts.append(f"dmg={damage:.0f}")
    if p1p1 > 0:
        parts.append(f"+1/+1={p1p1:.0f}")
    if attacking:
        parts.append("ATTACKING")
    if blocking:
        parts.append("BLOCKING")
    if kws:
        parts.append(f"[{', '.join(kws)}]")
    if playable_exile:
        parts.append("PLAYABLE")

    return " ".join(parts)


def format_zone(zone_data: np.ndarray, zone_name: str) -> list[str]:
    """Format all cards in a zone."""
    cards = []
    for i in range(zone_data.shape[0]):
        s = format_card(zone_data[i])
        if s:
            cards.append(f"    {s}")
    if not cards:
        return [f"  {zone_name}: (empty)"]
    return [f"  {zone_name}:"] + cards


def format_scalars(scalars: np.ndarray, label: str) -> str:
    """Format player scalar features."""
    life = int(scalars[0])
    hand = int(scalars[1])
    library = int(scalars[2])
    lands_played = int(scalars[3])
    max_lands = int(scalars[4])
    mana = []
    mana_names = ["W", "U", "B", "R", "G", "C"]
    for i in range(6):
        v = int(scalars[5 + i])
        if v > 0:
            mana.append(f"{v}{mana_names[i]}")
    mana_str = ", ".join(mana) if mana else "none"
    return f"  {label}: life={life} hand={hand} library={library} lands={lands_played}/{max_lands} mana=[{mana_str}]"


def format_stack(stack_data: np.ndarray) -> list[str]:
    """Format the stack."""
    entries = []
    for i in range(stack_data.shape[0]):
        src_id = int(stack_data[i, 0])
        if src_id <= 0:
            continue
        controller = "you" if int(stack_data[i, 1]) == 0 else "opp"
        tgt_id = int(stack_data[i, 2])
        tgt_player = stack_data[i, 3] > 0.5
        tgt_own = stack_data[i, 4] > 0.5
        is_creature = stack_data[i, 5] > 0.5

        src_name = card_name(src_id)
        tgt_str = ""
        if tgt_player:
            tgt_str = f" → {'you' if tgt_own else 'opponent'}"
        elif tgt_id > 0:
            tgt_str = f" → {card_name(tgt_id)}{'(own)' if tgt_own else ''}"
        entries.append(f"    [{controller}] {src_name}{tgt_str}")
    if not entries:
        return ["  Stack: (empty)"]
    return ["  Stack:"] + entries


def format_action(idx: int, features: np.ndarray, description: str) -> str:
    """Format a single action option."""
    src_id = int(features[0])
    is_pass = features[2] > 0.5
    tgt_player = features[3] > 0.5
    tgt_id = int(features[4])
    tgt_own = features[6] > 0.5
    power = features[7] * 20
    toughness = features[8] * 20
    cmc = features[9] * 10
    cost_tap = features[10] > 0.5
    cost_sac = features[11] > 0.5
    cost_life = features[12] > 0.5
    cost_exile_gy = features[13] > 0.5
    can_tgt_creatures = features[14] > 0.5
    can_tgt_players = features[15] > 0.5
    dmg = features[16] * 20
    from_exile = features[34] > 0.5 if len(features) > 34 else False

    if is_pass:
        return f"  [{idx}] Pass"

    src_name = card_name(src_id)
    parts = [f"  [{idx}] {src_name}"]

    if power > 0 or toughness > 0:
        parts.append(f"({power:.0f}/{toughness:.0f})")
    if cmc > 0:
        parts.append(f"cmc={cmc:.0f}")

    # Costs
    costs = []
    if cost_tap:
        costs.append("tap")
    if cost_sac:
        costs.append("sacrifice")
    if cost_life:
        costs.append("pay life")
    if cost_exile_gy:
        costs.append("exile from GY")
    if costs:
        parts.append(f"cost:[{','.join(costs)}]")

    # Target
    if tgt_player:
        parts.append(f"→ {'you' if tgt_own else 'opponent'}")
    elif tgt_id > 0:
        parts.append(f"→ {card_name(tgt_id)}{'(own)' if tgt_own else ''}")

    if dmg > 0:
        parts.append(f"dmg={dmg:.0f}")

    if from_exile:
        parts.append("[FROM EXILE]")

    # Description from server for reference
    parts.append(f"  // {description}")

    return " ".join(parts)


def display_observation(obs_proto, decision_point):
    """Display the full observation as the model sees it."""
    from forge_rl.env import ForgeRlEnv

    # Build obs using the env's logic
    env = ForgeRlEnv.__new__(ForgeRlEnv)
    obs = env._build_observation(obs_proto, decision_point)

    gi = obs["game_info"]
    phase_names = [
        "UNTAP", "UPKEEP", "DRAW", "MAIN1",
        "BEGIN_COMBAT", "DECLARE_ATTACKERS", "DECLARE_BLOCKERS",
        "FIRST_STRIKE_DAMAGE", "COMBAT_DAMAGE", "END_COMBAT",
        "MAIN2", "END_TURN", "CLEANUP",
        "?13", "?14", "?15", "?16", "?17", "?18"
    ]
    phase = phase_names[int(gi[1])] if 0 <= int(gi[1]) < len(phase_names) else f"phase_{int(gi[1])}"
    active = "you" if int(gi[2]) == 0 else "opp"
    priority = "you" if int(gi[3]) == 0 else "opp"

    print(f"\n{'='*60}")
    print(f"  Turn {int(gi[0])} | Phase: {phase} | Active: {active} | Priority: {priority}")
    print(f"  Mulligans: you={int(gi[4])} opp={int(gi[5])}")
    print(f"{'='*60}")

    # Players
    print(format_scalars(obs["agent_scalars"], "YOU"))
    print(format_scalars(obs["opponent_scalars"], "OPP"))

    # Zones
    print()
    for line in format_zone(obs["agent_hand"], "Your Hand"):
        print(line)
    for line in format_zone(obs["agent_battlefield"], "Your Battlefield"):
        print(line)
    for line in format_zone(obs["opponent_battlefield"], "Opp Battlefield"):
        print(line)
    for line in format_zone(obs["agent_graveyard"], "Your Graveyard"):
        print(line)
    for line in format_zone(obs["opponent_graveyard"], "Opp Graveyard"):
        print(line)
    for line in format_zone(obs["agent_exile"], "Your Exile"):
        print(line)
    for line in format_zone(obs["opponent_exile"], "Opp Exile"):
        print(line)
    for line in format_stack(obs["stack"]):
        print(line)

    # Decision
    dt = int(obs["decision_type"].argmax())
    dt_name = DECISION_TYPE_NAMES[dt] if dt < len(DECISION_TYPE_NAMES) else f"type_{dt}"
    print(f"\n  Decision: {dt_name}")
    if decision_point:
        print(f"  Prompt: {decision_point.prompt}")

    # Actions
    print(f"\n  Legal actions:")
    action_descs = {}
    if decision_point and decision_point.legal_actions:
        for action in decision_point.legal_actions:
            action_descs[action.index] = action.description

    af = obs["action_features"]
    mask = obs["action_mask"]
    for i in range(MAX_ACTIONS):
        if mask[i] > 0:
            desc = action_descs.get(i, "")
            print(format_action(i, af[i], desc))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Text-only MTG game with raw RL features")
    parser.add_argument("--deck-a", default="src/main/resources/decks/mono_red_pingers.dck",
                        help="Your deck")
    parser.add_argument("--deck-b", default="src/main/resources/decks/mono_red_pingers.dck",
                        help="Opponent deck (played by Forge AI)")
    parser.add_argument("--port", type=int, default=50099)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Register card names from both decks
    register_deck(args.deck_a)
    register_deck(args.deck_b)

    # Start server
    server_dir = str(Path(__file__).resolve().parent.parent / "forge-research")
    log_dir = "runs/text_game_debug"
    os.makedirs(log_dir, exist_ok=True)
    print("Starting game server...")
    procs = start_servers(1, args.port, server_dir,
                          deck_paths=[args.deck_a, args.deck_b], log_dir=log_dir)
    atexit.register(stop_servers, procs)
    print(f"Server log: {log_dir}/server_{args.port}.log")

    client = ForgeRlClient(port=args.port)

    print("\n" + "="*60)
    print("  TEXT-ONLY MTG — You see EXACTLY what the model sees.")
    print("  No hidden info, no card art, no flavor text.")
    print("  Just features and numbers. Can you win?")
    print("="*60)

    # Reset game
    resp = client.reset(
        deck_path_a=args.deck_a,
        deck_path_b=args.deck_b,
        agent_player_index=0,
        seed=args.seed,
    )

    obs_proto = resp.observation
    dp = resp.decision_point
    terminated = False

    while not terminated:
        if dp is None or len(dp.legal_actions) == 0:
            break

        display_observation(obs_proto, dp)

        # Get player input
        legal = [a.index for a in dp.legal_actions]
        while True:
            try:
                choice = input(f"\n  Your choice {legal}: ").strip()
                if choice.lower() in ('q', 'quit'):
                    print("Quitting.")
                    return
                choice = int(choice)
                if choice in legal:
                    break
                print(f"  Invalid. Choose from: {legal}")
            except (ValueError, EOFError):
                print(f"  Enter a number from: {legal}")

        resp = client.step(action_index=choice)
        terminated = resp.terminated
        obs_proto = resp.observation
        dp = resp.decision_point if not terminated else None

        if terminated and not resp.HasField("game_result"):
            print(f"\n  [DEBUG] Game terminated without game_result! reward={resp.reward}")
            print(f"  [DEBUG] decision_point present: {dp is not None}")
            break

        if resp.HasField("game_result"):
            gr = resp.game_result
            winner = "YOU WIN!" if gr.winner_index == 0 else "YOU LOSE."
            print(f"\n{'='*60}")
            print(f"  GAME OVER — {winner}")
            print(f"  Turns played: {gr.turns_played}")
            print(f"  Final life: you={obs_proto.agent_player.life} opp={obs_proto.opponent_player.life}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
