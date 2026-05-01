"""
Test the PPO agent on the Balatro environment.
"""
import numpy as np
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from itertools import combinations
from ppoagent import RANKS, BalatroPLayEnv, ScoreCallback  


agent=PPO.load("balatro_ppo")
# agent = PPO.load("balatro_penalized_discard_ppo")

URL = "http://127.0.0.1:12346"

# STATS AS OF 2024-06-01
HAND_SIZE = 8
MAX_HANDS = 4
MAX_DISCARDS = 4
TARGET_SCORE = 300


def rpc(method: str, params: dict | None = None) -> dict:
    resp = requests.post(
        URL,
        json={"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1},
    )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"]["message"])
    return data["result"]


_subsets = [
    list(combo)
    for size in range(1, 6)
    for combo in combinations(range(HAND_SIZE), size)
]
_n_subsets = len(_subsets)


# Create mappings for action decoding
def parse_card(card):
    rank_map = {r: i for i, r in enumerate(RANKS)}
    suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
    rank_str = card['value']['rank']
    suit_str = card['value']['suit']
    return rank_map.get(rank_str, 0), suit_map.get(suit_str, 0)


def build_obs(hand_cards, score, hands_remaining, discards_remaining):
    """
    Build the flat observation vector the trained agent expects.
    Pads to HAND_SIZE if fewer cards are visible.
    """
    parsed = [parse_card(c) for c in hand_cards]

    # Pad with zeros if hand is smaller than expected
    while len(parsed) < HAND_SIZE:
        parsed.append((0, 0))

    ranks = np.array([c[0] / 12.0 for c in parsed], dtype=np.float32)
    suits = np.array([c[1] / 3.0  for c in parsed], dtype=np.float32)

    return np.concatenate([
        ranks, suits,
        [
            min(score / TARGET_SCORE, 1.0),
            hands_remaining / MAX_HANDS,
            discards_remaining / MAX_DISCARDS,
        ]
    ])

def decode_action(action, hand_cards):
    """
    Convert the agent's integer action into BalatroBot API arguments.
    Returns (is_discard, card_indices_in_real_hand).
    """
    is_discard = action >= _n_subsets
    subset_indices = _subsets[action % _n_subsets]

    # Clamp indices to actual hand size (safety)
    valid_indices = [i for i in subset_indices if i < len(hand_cards)]
    return is_discard, valid_indices


def get_hand_state(state):
    return (
        state["hand"]["cards"],
        state["round"]["chips"],
        state["round"]["hands_left"],
        state["round"]["discards_left"],
    )


def play_game():
    state = rpc("start", {"deck": "RED", "stake": "WHITE"})
    print(f"Game started | seed={state.get('seed', '?')}")
    print(f"Raw start response: {state}")

    while state.get("state") != "GAME_OVER":
        curr = state.get("state")

        # ── phases we skip for now ──────────────────────────────────────────
        if curr == "BLIND_SELECT":
            state = rpc("select")   # just select whatever is offered
            continue

        if curr == "SHOP":
            state = rpc("next_round")               # skip shop entirely
            continue

        if curr == "ROUND_EVAL":
            state = rpc("cash_out")               # collect end-of-round cash
            continue

        # ── hand-playing phase ──────────────────────────────────────────────
        if curr == "SELECTING_HAND":
            hand_cards, score, hands_left, discards_left = get_hand_state(state)

            obs = build_obs(hand_cards, score, hands_left, discards_left)

            # SB3 expects a batched obs
            action, _ = agent.predict(obs[np.newaxis, :], deterministic=True)
            action = int(action[0])

            is_discard, indices = decode_action(action, hand_cards)

            # Guard: if agent wants to discard but none left, force a play
            if is_discard and discards_left <= 0:
                is_discard = False

            # Guard: always play at least 1 card
            if not indices:
                indices = [0]

            if is_discard:
                print(f"  DISCARD indices={indices}")
                state = rpc("discard", {"cards": indices})
            else:
                print(f"  PLAY    indices={indices}")
                state = rpc("play", {"cards": indices})

            continue

        # ── fallback for any state we don't handle yet ──────────────────────
        print(f"  Unhandled state: {curr} — skipping")
        state = rpc("skip")   # or whatever BalatroBot uses as a no-op
        continue

    print(f"Game over | final state: {state}")


if __name__ == "__main__":
    play_game()

