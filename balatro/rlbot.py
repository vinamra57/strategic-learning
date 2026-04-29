# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
# ]
# ///

"""
Reinforcement learning bot for Balatro using tabular Q-learning.

The greedy bot in bot.py always plays the best current hand immediately.
This RL bot learns *when to discard* (try to improve the hand) vs *play now*,
and also whether to skip blinds.

State features for SELECTING_HAND:
  (ante_bucket, hands_left, discards_left, best_hand_rank, chip_progress)

Actions:
  - "play"        : play the best 5-card hand immediately
  - "discard"     : discard cards not contributing to the best hand (deadwood)
  - "discard_all" : discard 5 lowest-value cards (very aggressive rebuild)

Learning is Monte Carlo: collect (state, action) trajectory each game, then
backpropagate the final discounted reward at game end.

Q-table and epsilon are persisted to qtable.json after every game so learning
accumulates across many runs.
"""

import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import requests

URL = "http://127.0.0.1:12346"
QTABLE_PATH = Path(__file__).parent / "qtable.json"
STATS_PATH = Path(__file__).parent / "rl_stats.json"

RANK_ORDER = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

# Numeric rank for each hand type (higher = better)
HAND_RANKS = {
    "High Card": 0, "Pair": 1, "Two Pair": 2, "Three of a Kind": 3,
    "Straight": 4, "Flush": 5, "Full House": 6, "Four of a Kind": 7,
    "Straight Flush": 8, "Five of a Kind": 9, "Flush Five": 10, "Flush House": 10,
}

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
ALPHA = 0.1           # Q-learning rate
GAMMA = 0.90          # reward discount per step
EPSILON_START = 0.40  # initial exploration probability
EPSILON_MIN = 0.05    # floor for epsilon
EPSILON_DECAY = 0.992 # multiplicative decay applied after each game

# ---------------------------------------------------------------------------
# Card model (mirrors bot.py)
# ---------------------------------------------------------------------------
hand_values: dict[str, tuple[int, int]] = {}


class Card:
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        self.value = int(rank) if rank.isdigit() else (11 if rank == "A" else 10)
        self.rank_order = RANK_ORDER[rank]

    def __repr__(self) -> str:
        return f"{self.rank}{self.suit}"


def parse_cards(cards: list[dict]) -> list[Card]:
    return [Card(c["value"]["rank"], c["value"]["suit"]) for c in cards]


def is_straight(cards: list[Card]) -> bool:
    ranks = sorted(c.rank_order for c in cards)
    return ranks == list(range(ranks[0], ranks[0] + 5))


def is_flush(cards: list[Card]) -> bool:
    return len({c.suit for c in cards}) == 1


def pick_hand(card: list[Card] | tuple[Card, ...]) -> str:
    """Identify hand type for a 5-card combo."""
    card = sorted(card, key=lambda c: c.rank_order)
    if (card[0].rank == card[1].rank == card[2].rank == card[3].rank == card[4].rank):
        return "Flush Five" if is_flush(card) else "Five of a Kind"
    if (
        (card[0].rank == card[1].rank == card[2].rank and card[3].rank == card[4].rank)
        or (card[0].rank == card[1].rank and card[2].rank == card[3].rank == card[4].rank)
    ):
        return "Flush House" if is_flush(card) else "Full House"
    if card[1].rank == card[2].rank == card[3].rank and (
        card[0].rank == card[1].rank or card[3].rank == card[4].rank
    ):
        return "Four of a Kind"
    if is_flush(card):
        return "Straight Flush" if is_straight(card) else "Flush"
    if is_straight(card):
        return "Straight"
    if (
        card[0].rank == card[1].rank == card[2].rank
        or card[1].rank == card[2].rank == card[3].rank
        or card[2].rank == card[3].rank == card[4].rank
    ):
        return "Three of a Kind"
    if (
        (card[0].rank == card[1].rank and card[2].rank == card[3].rank)
        or (card[0].rank == card[1].rank and card[3].rank == card[4].rank)
        or (card[1].rank == card[2].rank and card[3].rank == card[4].rank)
    ):
        return "Two Pair"
    if (
        card[0].rank == card[1].rank
        or card[1].rank == card[2].rank
        or card[2].rank == card[3].rank
        or card[3].rank == card[4].rank
    ):
        return "Pair"
    return "High Card"


def score_hand(hand: str, cards: list[Card]) -> int:
    chips, mult = hand_values[hand]
    ranks = Counter(c.rank for c in cards)
    if hand in ("Straight Flush", "Flush Five", "Full House", "Flush House",
                "Straight", "Flush", "Five of a Kind"):
        chips += sum(c.value for c in cards)
    elif hand == "Four of a Kind":
        chips += sum(c.value for c in cards if ranks[c.rank] == 4)
    elif hand == "Three of a Kind":
        chips += sum(c.value for c in cards if ranks[c.rank] == 3)
    elif hand in ("Two Pair", "Pair"):
        chips += sum(c.value for c in cards if ranks[c.rank] == 2)
    else:
        chips += max(c.value for c in cards)
    return chips * mult


def best_hand_from(cards: list[Card]) -> tuple[int, tuple[Card, ...], str]:
    """Return (score, best_combo, hand_type) for the best 5-card hand in cards."""
    best_score, best_combo, best_type = 0, None, "High Card"
    for combo in itertools.combinations(cards, 5):
        hand_type = pick_hand(combo)
        score = score_hand(hand_type, combo)
        if score > best_score:
            best_score, best_combo, best_type = score, combo, hand_type
    return best_score, best_combo, best_type


def initialize_hands(init: dict) -> None:
    for hand, data in init["hands"].items():
        hand_values[hand] = (data["chips"], data["mult"])


# ---------------------------------------------------------------------------
# Discard helpers
# ---------------------------------------------------------------------------

def cards_to_discard(cards: list[Card], action: str) -> list[int]:
    """
    Return indices of cards to discard for the given action.

    "discard"     — remove cards NOT contributing to the best hand (deadwood).
    "discard_all" — remove the 5 lowest-value cards (aggressive rebuild).
    """
    if action == "discard":
        _, best_combo, _ = best_hand_from(cards)
        best_ids = {id(c) for c in best_combo}
        deadwood = [i for i, c in enumerate(cards) if id(c) not in best_ids]
        return deadwood[:5]

    # discard_all: lowest-value cards first
    sorted_idx = sorted(range(len(cards)), key=lambda i: cards[i].value)
    return sorted_idx[:5]


# ---------------------------------------------------------------------------
# State feature extraction
# ---------------------------------------------------------------------------

def _round(state: dict) -> dict:
    """Return state["round"] dict, or empty dict if missing."""
    return state.get("round") or {}


def _current_blind_score(state: dict) -> int:
    """
    Find the blind with status CURRENT and return its chip target.
    state["blinds"] is a dict of blind objects keyed by type name.
    Falls back to 300 if nothing found.
    """
    blinds = state.get("blinds") or {}
    for blind in blinds.values():
        if isinstance(blind, dict) and blind.get("status") == "CURRENT":
            return blind.get("score", 300)
    return 300


def hand_state(state: dict, cards: list[Card]) -> tuple:
    """
    Discretised state for SELECTING_HAND:
      (ante_bucket, hands_left, discards_left, best_hand_rank, progress_bucket)

    Field paths per API docs:
      state["ante_num"]
      state["round"]["hands_left"]
      state["round"]["discards_left"]
      state["round"]["chips"]          — chips scored so far this round
      current blind score via state["blinds"][*]["score"] where status==CURRENT
    """
    ante = min(state.get("ante_num", 1), 8)
    ante_bucket = min((ante - 1) // 2, 3)  # 0..3 (pairs of antes)

    rnd = _round(state)
    hands_left = min(rnd.get("hands_left", 4), 8)
    discards_left = min(rnd.get("discards_left", 3), 5)

    _, _, hand_type = best_hand_from(cards)
    hand_rank = HAND_RANKS.get(hand_type, 0)

    chips_scored = rnd.get("chips", 0)
    chips_needed = max(_current_blind_score(state), 1)
    progress = min(4, int(chips_scored / chips_needed * 5))  # 0..4

    return (ante_bucket, hands_left, discards_left, hand_rank, progress)


def blind_state(state: dict) -> tuple:
    """State for BLIND_SELECT: ante bucket only."""
    ante = min(state.get("ante_num", 1), 8)
    return (min((ante - 1) // 2, 3),)


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class QAgent:
    HAND_ACTIONS = ("play", "discard", "discard_all")
    BLIND_ACTIONS = ("select", "skip")

    def __init__(self) -> None:
        self.q: dict[str, float] = defaultdict(float)
        self.epsilon = EPSILON_START
        self.game_count = 0
        # trajectory: list of (state_type, features, action) recorded this game
        self._traj: list[tuple[str, tuple, str]] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if QTABLE_PATH.exists():
            raw = json.loads(QTABLE_PATH.read_text())
            self.q = defaultdict(float, raw.get("q_table", {}))
            self.epsilon = raw.get("epsilon", EPSILON_START)
            self.game_count = raw.get("game_count", 0)
            print(
                f"[RL] Loaded Q-table ({len(self.q)} entries) | "
                f"game #{self.game_count + 1} | ε={self.epsilon:.3f}"
            )
        else:
            print("[RL] No Q-table found — starting fresh.")

    def _save(self) -> None:
        QTABLE_PATH.write_text(
            json.dumps(
                {"q_table": dict(self.q), "epsilon": self.epsilon, "game_count": self.game_count},
                indent=2,
            )
        )

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------

    def _key(self, stype: str, feat: tuple, action: str) -> str:
        return f"{stype}|{feat}|{action}"

    def _q_val(self, stype: str, feat: tuple, action: str) -> float:
        return self.q[self._key(stype, feat, action)]

    def _best(self, stype: str, feat: tuple, actions: tuple[str, ...]) -> str:
        return max(actions, key=lambda a: self._q_val(stype, feat, a))

    def choose(self, stype: str, feat: tuple, actions: tuple[str, ...]) -> str:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self._best(stype, feat, actions)

    def record(self, stype: str, feat: tuple, action: str) -> None:
        self._traj.append((stype, feat, action))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def end_game(self, reward: float) -> None:
        """
        Monte Carlo update: walk trajectory backward and apply TD-style updates
        using discounted return G starting from the final reward.
        """
        G = reward
        for stype, feat, action in reversed(self._traj):
            key = self._key(stype, feat, action)
            # Incremental mean toward G (acts like every-visit MC with step-size α)
            self.q[key] += ALPHA * (G - self.q[key])
            G *= GAMMA

        self._traj = []
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.game_count += 1
        self._save()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def top_actions(self, n: int = 10) -> None:
        """Print the highest-confidence Q-values learned so far."""
        if not self.q:
            print("[RL] Q-table is empty.")
            return
        ranked = sorted(self.q.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
        print("[RL] Top Q-values:")
        for key, val in ranked:
            print(f"  {val:+.2f}  {key}")


# ---------------------------------------------------------------------------
# Game stats tracker
# ---------------------------------------------------------------------------

class StatsTracker:
    def __init__(self) -> None:
        self.wins = 0
        self.losses = 0
        self.max_ante = 0
        self._load()

    def _load(self) -> None:
        if STATS_PATH.exists():
            d = json.loads(STATS_PATH.read_text())
            self.wins = d.get("wins", 0)
            self.losses = d.get("losses", 0)
            self.max_ante = d.get("max_ante", 0)

    def record(self, won: bool, ante: int) -> None:
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.max_ante = max(self.max_ante, ante)
        STATS_PATH.write_text(
            json.dumps({"wins": self.wins, "losses": self.losses, "max_ante": self.max_ante})
        )

    def summary(self) -> str:
        total = self.wins + self.losses
        rate = self.wins / total * 100 if total else 0
        return (
            f"[Stats] W={self.wins} L={self.losses} "
            f"({rate:.1f}% win rate) | best ante={self.max_ante}"
        )


# ---------------------------------------------------------------------------
# RPC
# ---------------------------------------------------------------------------

def rpc(method: str, params: dict | None = None) -> dict:
    resp = requests.post(
        URL,
        json={"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1},
    )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"]["message"])
    return data["result"]


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def play_game(agent: QAgent, stats: StatsTracker) -> float:
    """
    Run one full game with the RL agent.
    Returns the total reward for the game (used for Q-table update).
    """
    init = rpc("menu")
    initialize_hands(init)
    state = rpc("start", {"deck": "RED", "stake": "WHITE"})
    print(f"\n[Game {agent.game_count + 1}] seed={state['seed']} | ε={agent.epsilon:.3f}")

    total_reward = 0.0
    rounds_cleared = 0

    while state["state"] != "GAME_OVER":
        match state["state"]:

            case "BLIND_SELECT":
                feat = blind_state(state)
                action = agent.choose("blind", feat, QAgent.BLIND_ACTIONS)
                agent.record("blind", feat, action)

                if action == "skip":
                    try:
                        state = rpc("skip")
                    except RuntimeError:
                        # Skip not available (e.g. boss blind) — fall back
                        state = rpc("select")
                else:
                    state = rpc("select")

            case "SELECTING_HAND":
                cards = parse_cards(state["hand"]["cards"])
                feat = hand_state(state, cards)
                discards_left = feat[2]  # extracted in hand_state

                # Only offer discard actions if we actually have discards
                if discards_left > 0:
                    valid = QAgent.HAND_ACTIONS
                else:
                    valid = ("play",)

                action = agent.choose("hand", feat, valid)
                agent.record("hand", feat, action)

                if action == "play":
                    _, best_combo, _ = best_hand_from(cards)
                    indices = [cards.index(c) for c in best_combo]
                    state = rpc("play", {"cards": indices})
                else:
                    indices = cards_to_discard(cards, action)
                    try:
                        state = rpc("discard", {"cards": indices})
                    except RuntimeError:
                        # Discard failed (e.g. no discards left) — play instead
                        _, best_combo, _ = best_hand_from(cards)
                        indices = [cards.index(c) for c in best_combo]
                        state = rpc("play", {"cards": indices})

            case "ROUND_EVAL":
                ante = state.get("ante_num", 1)
                round_reward = float(ante) * 3.0   # more reward for clearing higher antes
                total_reward += round_reward
                rounds_cleared += 1
                print(
                    f"  Round cleared (ante {ante}) +{round_reward:.0f} "
                    f"[total={total_reward:.0f}]"
                )
                state = rpc("cash_out")

            case "SHOP":
                state = rpc("next_round")

            case "SMODS_BOOSTER_OPENED":
                # A booster pack is open. We always skip.
                # Some packs (e.g. Mega packs) allow multiple picks, so the
                # state may stay as SMODS_BOOSTER_OPENED after one skip — loop
                # until we're fully out to avoid stale pack cards carrying over.
                max_skips = 10
                for _ in range(max_skips):
                    state = rpc("pack", {"skip": True})
                    if state["state"] != "SMODS_BOOSTER_OPENED":
                        break
                # Explicitly re-fetch a clean gamestate so no pack card data
                # bleeds into subsequent state reads (known stale-state behaviour
                # in balatrobot — not tracked as an upstream issue as of Apr 2026).
                state = rpc("gamestate")

            case _:
                state = rpc("gamestate")

    won = state.get("won", False)
    ante = state.get("ante_num", 1)

    if won:
        total_reward += 200.0
        print(f"  *** WIN *** ante={ante} | reward={total_reward:.1f}")
    else:
        total_reward -= 15.0
        print(f"  LOSS at ante={ante} round={state.get('round_num', '?')} | reward={total_reward:.1f}")

    stats.record(won, ante)
    return total_reward


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    agent = QAgent()
    stats = StatsTracker()

    print(stats.summary())

    try:
        while True:
            reward = play_game(agent, stats)
            agent.end_game(reward)
            print(
                f"[RL] Game {agent.game_count} complete | "
                f"reward={reward:.1f} | ε={agent.epsilon:.3f} | "
                f"Q-entries={len(agent.q)}"
            )
            print(stats.summary())
            agent.top_actions(5)
    except KeyboardInterrupt:
        print("\n[RL] Stopped. Q-table saved.")
        agent.top_actions(10)


if __name__ == "__main__":
    main()
