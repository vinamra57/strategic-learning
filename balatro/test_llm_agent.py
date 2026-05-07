"""
Live runner: PPO plays hands, LLM advisor handles the shop, packs, and rerolls.

Loop sketch:
  BLIND_SELECT       → select  (RL agent doesn't decide blinds yet)
  SELECTING_HAND     → PPO picks 1-5 cards to play / discard
  ROUND_EVAL         → cash_out
  SHOP               → LLM advisor loops:  ask → execute → re-fetch
                       → next_round on skip
  SMODS_BOOSTER_OPENED → LLM picks card from pack (or skip)

Hand-history is accumulated across the whole game and fed to every
advisor call so it can adapt picks to what the agent actually plays.

Logs:
  logs/shop_advisor.jsonl  — every advisor decision (auto, from advisor)
  logs/run.jsonl           — high-level per-step game events (this file)
"""
from __future__ import annotations
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import requests
from stable_baselines3 import PPO

from llm.advisor import ShopAdvisor
from llm.actions import BuyShopItem, BuyVoucher, BuyPack, Reroll, Skip
from llm.state import HandHistory, parse_shop_context
from ppoagent import RANKS, evaluate_hand


URL = "http://127.0.0.1:12346"
HAND_SIZE = 8
MAX_HANDS = 4
MAX_DISCARDS = 4
TARGET_SCORE = 300       # ante 1 small blind

# Log paths — overridable via env so the many_games runner can route
# each game's logs into a dedicated per-seed folder.
RUN_LOG = Path(os.environ.get("BALATRO_RUN_LOG", "logs/run.jsonl"))
SHOP_LOG = os.environ.get("BALATRO_SHOP_LOG", "logs/shop_advisor.jsonl")
RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
Path(SHOP_LOG).parent.mkdir(parents=True, exist_ok=True)

_SUBSETS = [
    list(combo)
    for size in range(1, 6)
    for combo in combinations(range(HAND_SIZE), size)
]
_N_SUBSETS = len(_SUBSETS)


# ─── RPC helpers ──────────────────────────────────────────────────────────

def rpc(method: str, params: dict | None = None) -> dict:
    resp = requests.post(
        URL,
        json={"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1},
        timeout=10,
    )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"]["message"])
    return data["result"]


# ─── Card / observation helpers (for PPO) ─────────────────────────────────

def _parse_card(card: dict) -> tuple[int, int]:
    rank_map = {r: i for i, r in enumerate(RANKS)}
    suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}
    rank_str = card["value"]["rank"]
    suit_str = card["value"]["suit"]
    return rank_map.get(rank_str, 0), suit_map.get(suit_str, 0)


def _build_obs(hand_cards, score, hands_left, discards_left) -> np.ndarray:
    parsed = [_parse_card(c) for c in hand_cards]
    while len(parsed) < HAND_SIZE:
        parsed.append((0, 0))
    ranks = np.array([c[0] / 12.0 for c in parsed], dtype=np.float32)
    suits = np.array([c[1] / 3.0 for c in parsed], dtype=np.float32)
    return np.concatenate([
        ranks, suits,
        [
            min(score / TARGET_SCORE, 1.0),
            hands_left / MAX_HANDS,
            discards_left / MAX_DISCARDS,
        ],
    ])


def _decode_action(action: int, hand_cards: list) -> tuple[bool, list[int]]:
    is_discard = action >= _N_SUBSETS
    subset = _SUBSETS[action % _N_SUBSETS]
    valid = [i for i in subset if i < len(hand_cards)]
    return is_discard, valid


# ─── Logging ──────────────────────────────────────────────────────────────

def log_event(event: str, payload: dict) -> None:
    """`event` is the high-level event name; payload may contain its own
    `kind` field (e.g. for action.kind) without clobbering."""
    entry = {"ts": time.time(), "event": event, "data": payload}
    with RUN_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── Phase handlers ───────────────────────────────────────────────────────

def handle_selecting_hand(
    state: dict, agent: PPO, history: HandHistory,
) -> dict:
    rnd = state.get("round") or {}
    hand_cards = (state.get("hand") or {}).get("cards", [])
    score = rnd.get("chips", 0)
    hands_left = rnd.get("hands_left", 0)
    discards_left = rnd.get("discards_left", 0)

    obs = _build_obs(hand_cards, score, hands_left, discards_left)
    action, _ = agent.predict(obs[np.newaxis, :], deterministic=True)
    action = int(action[0])
    is_discard, indices = _decode_action(action, hand_cards)

    # Guards: never discard with 0 left, always play at least 1 card
    if is_discard and discards_left <= 0:
        is_discard = False
    if not indices:
        indices = [0]

    if is_discard:
        history.record_discard()
        log_event("discard", {"indices": indices})
        return rpc("discard", {"cards": indices})

    # PLAY: also figure out the resulting hand type/score so we can record it
    selected = [_parse_card(hand_cards[i]) for i in indices]
    if len(selected) == 5:
        hand_name, hand_score = evaluate_hand(selected)
    else:
        # Sub-5 hands evaluated by the same logic
        hand_name, hand_score = evaluate_hand(selected)
    history.record_play(hand_name, hand_score)
    log_event("play", {"indices": indices, "hand": hand_name, "score": hand_score})
    return rpc("play", {"cards": indices})


def handle_shop(state: dict, advisor: ShopAdvisor, history: HandHistory) -> dict:
    """Loop: ask advisor → execute → re-fetch → repeat until skip / stuck."""
    max_iterations = 12   # safety cap — shouldn't fire in practice
    last_state_sig = None
    for _ in range(max_iterations):
        ctx = parse_shop_context(state, history)
        action = advisor.decide_shop(ctx)
        log_event("shop_decision", {
            "kind": action.kind,
            "index": getattr(action, "index", None),
            "reason": getattr(action, "reason", ""),
            "money": ctx.money,
            "ante": ctx.ante,
        })

        if isinstance(action, Skip):
            return rpc("next_round")

        try:
            if isinstance(action, BuyShopItem):
                state = rpc("buy", {"card": action.index})
            elif isinstance(action, BuyVoucher):
                state = rpc("buy", {"voucher": action.index})
            elif isinstance(action, BuyPack):
                state = rpc("buy", {"pack": action.index})
            elif isinstance(action, Reroll):
                state = rpc("reroll")
        except RuntimeError as e:
            log_event("shop_rpc_error", {"error": str(e), "kind": action.kind})
            # If the RPC rejected our action, leave shop to avoid infinite loop
            return rpc("next_round")

        # Pack opens immediately switch state
        if state.get("state") == "SMODS_BOOSTER_OPENED":
            state = handle_pack_open(state, advisor, history)

        # Detect stuck: if state unchanged across iterations, bail.
        sig = json.dumps({
            "money": state.get("money"),
            "shop_count": (state.get("shop") or {}).get("count"),
            "voucher_count": (state.get("vouchers") or {}).get("count"),
            "pack_count": (state.get("packs") or {}).get("count"),
            "phase": state.get("state"),
        }, sort_keys=True)
        if sig == last_state_sig:
            log_event("shop_stuck_break", {"sig": sig})
            return rpc("next_round")
        last_state_sig = sig

        # If the action moved us out of SHOP entirely, we're done
        if state.get("state") != "SHOP":
            return state

    log_event("shop_iter_cap", {"reason": "max iterations reached"})
    return rpc("next_round")


def handle_pack_open(state: dict, advisor: ShopAdvisor, history: HandHistory) -> dict:
    """Pick from / skip an opened booster pack. Loops until pack is closed."""
    max_picks = 5
    for _ in range(max_picks):
        if state.get("state") != "SMODS_BOOSTER_OPENED":
            return state
        pack = state.get("pack") or {}
        cards = pack.get("cards", [])
        limit = pack.get("limit", 1)
        label = pack.get("label", "Pack")
        ctx = parse_shop_context(state, history)
        decision = advisor.decide_pack(label, cards, limit, ctx)
        log_event("pack_pick", {
            "pack": label,
            "index": decision.index,
            "reason": decision.reason,
        })
        try:
            if decision.index is None:
                state = rpc("pack", {"skip": True})
                # If we're still in pack state, force skip continuously
                while state.get("state") == "SMODS_BOOSTER_OPENED":
                    state = rpc("pack", {"skip": True})
                return state
            else:
                state = rpc("pack", {"card": decision.index})
        except RuntimeError as e:
            log_event("pack_rpc_error", {"error": str(e)})
            return rpc("gamestate")
    return rpc("gamestate")


# ─── Main game loop ───────────────────────────────────────────────────────

def play_game(model_path: str = "balatro_ppo",
              llm_model: str = "qwen3:4b-instruct") -> dict:
    agent = PPO.load(model_path)
    advisor = ShopAdvisor(model=llm_model, log_path=SHOP_LOG)
    history = HandHistory()

    state = rpc("start", {"deck": "RED", "stake": "WHITE"})
    print(f"Game started | seed={state.get('seed')}")
    log_event("start", {"seed": state.get("seed")})

    step = 0
    while state.get("state") != "GAME_OVER":
        phase = state.get("state")
        step += 1

        if phase == "BLIND_SELECT":
            state = rpc("select")
            continue
        if phase == "ROUND_EVAL":
            state = rpc("cash_out")
            continue
        if phase == "SHOP":
            state = handle_shop(state, advisor, history)
            continue
        if phase == "SMODS_BOOSTER_OPENED":
            state = handle_pack_open(state, advisor, history)
            continue
        if phase == "SELECTING_HAND":
            state = handle_selecting_hand(state, agent, history)
            continue

        # Unknown / passthrough
        state = rpc("gamestate")

    won = state.get("won", False)
    ante = state.get("ante_num", 1)
    rnd = state.get("round_num", 1)
    print(f"Game over | won={won} ante={ante} round={rnd} | "
          f"hands_played={history.total_plays}, "
          f"top={history.top_hands(3)}, "
          f"llm_calls={advisor.llm_calls}")
    log_event("game_over", {
        "won": won, "ante": ante, "round": rnd,
        "hands_played": history.total_plays,
        "top_hands": history.top_hands(5),
        "llm_calls": advisor.llm_calls,
    })
    return {"won": won, "ante": ante, "round": rnd, "history": history}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="balatro_ppo", help="PPO checkpoint")
    ap.add_argument("--llm", default="qwen3:4b-instruct", help="Ollama model tag")
    ap.add_argument("--games", type=int, default=1)
    args = ap.parse_args()

    for g in range(args.games):
        print(f"\n=== Game {g+1}/{args.games} ===")
        play_game(args.model, args.llm)
