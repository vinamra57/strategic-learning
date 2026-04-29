# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
# ]
# ///

import itertools
import random
from collections import Counter
import requests

URL = "http://127.0.0.1:12346"

RANK_ORDER = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
              "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

hand_values = {}


class Card:
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        if rank.isdigit():
            self.value = int(rank)
        elif rank == "A":
            self.value = 11
        else:
            self.value = 10
        self.rank_order = RANK_ORDER[self.rank]
    
    def __repr__(self):
        return f"{self.rank}{self.suit}"

# ---------------------------------------------------------------------------
# Card parsing
# ---------------------------------------------------------------------------

def parse_cards(cards: list[dict]) -> list[Card]:
    """Parse a list of card dicts into Card objects."""
    return [Card(c["value"]["rank"], c["value"]["suit"]) for c in cards]

def is_straight(cards: list[Card]) -> bool:
    """Check if a hand of cards is a straight."""
    ranks = sorted(c.rank_order for c in cards)
    return ranks == list(range(ranks[0], ranks[0] + 5))

def is_flush(cards: list[Card]) -> bool:
    """Check if a hand of cards is a flush."""
    suits = set(c.suit for c in cards)
    return len(suits) == 1

def pick_hand(card: list[Card]) -> str:
    """Pick the best hand from a list of cards."""
    if (card[0].rank == card[1].rank and card[1].rank == card[2].rank and card[2].rank == card[3].rank and card[3].rank == card[4].rank):
        if is_flush(card):
            return "Flush Five"
        else:
            return "Five of a Kind"
    elif (card[0].rank == card[1].rank and card[1].rank == card[2].rank and card[3].rank == card[4].rank):
        if is_flush(card):
            return "Flush House"
        else:
            return "Full House"
    elif card[1].rank == card[2].rank and card[2].rank == card[3].rank and (card[0].rank == card[1].rank or card[3].rank == card[4].rank):
        return "Four of a Kind"
    elif is_flush(card):
        if is_straight(card):
            return "Straight Flush"
        else:
            return "Flush"
    elif is_straight(card):
        return "Straight"
    elif (card[0].rank == card[1].rank and card[1].rank == card[2].rank) or (card[1].rank == card[2].rank and card[2].rank == card[3].rank) or (card[2].rank == card[3].rank and card[3].rank == card[4].rank):
        return "Three of a Kind"
    elif (card[0].rank == card[1].rank and card[2].rank == card[3].rank) or (card[0].rank == card[1].rank and card[3].rank == card[4].rank) or (card[1].rank == card[2].rank and card[3].rank == card[4].rank):
        return "Two Pair"
    elif card[0].rank == card[1].rank or card[1].rank == card[2].rank or card[2].rank == card[3].rank or card[3].rank == card[4].rank:
        return "Pair"
    else:
        return "High Card"

def score_hand(hand: str, cards: list[Card]) -> int:
    """Score a hand based on its type and the ranks of the cards."""
    chips = hand_values[hand][0]
    mult = hand_values[hand][1]

    ranks = Counter(c.rank for c in cards)

    if hand in ["Straight Flush", "Flush Five", "Full House", "Flush House", "Straight", "Flush", "Five of a Kind"]:
        chips += sum(c.value for c in cards)
    elif hand == "Four of a Kind":
        chips += sum(c.value if ranks[c.rank] == 4 else 0 for c in cards)
    elif hand == "Three of a Kind":
        chips += sum(c.value if ranks[c.rank] == 3 else 0 for c in cards)
    elif hand in ["Two Pair", "Pair"]:
        chips += sum(c.value if ranks[c.rank] == 2 else 0 for c in cards)
    else:
        chips += max(c.value for c in cards)
    return chips * mult

def best_hand(cards: list[Card]) -> tuple[int, list[Card]]:
    """Determine the best hand from a list of cards and return its score."""
    best_score = 0
    best_hand = None

    for combo in itertools.combinations(cards, 5):
        hand_type = pick_hand(combo)
        score = score_hand(hand_type, combo)
        if score > best_score:
            best_score = score
            best_hand = combo

    return best_score, best_hand, pick_hand(best_hand)

def initialize_hands(init: dict):
    """Initialize the bot's hand selection based on the initial menu state."""
    # For now, just select the first 5 cards for each hand
    for hand, hand_data in init["hands"].items():
        hand_values[hand] = (hand_data["chips"], hand_data["mult"])

# ---------------------------------------------------------------------------
# RPC
# ---------------------------------------------------------------------------

def rpc(method: str, params: dict = {}) -> dict:
    response = requests.post(URL, json={
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    })
    data = response.json()
    if "error" in data:
        raise Exception(data["error"]["message"])

    return data["result"]

# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def play_game():
    init = rpc("menu")
    initialize_hands(init)
    state = rpc("start", {"deck": "RED", "stake": "WHITE"})
    print(f"Started game with seed: {state['seed']}")

    while state["state"] != "GAME_OVER":
        match state["state"]:
            case "BLIND_SELECT":
                state = rpc("select")

            case "SELECTING_HAND":
                cards = parse_cards(state["hand"]["cards"])
                state = rpc("play", {"cards": [cards.index(c) for c in best_hand(cards)[1]]})

            case "ROUND_EVAL":
                state = rpc("cash_out")

            case "SHOP":
                state = rpc("next_round")

            case _:
                state = rpc("gamestate")

    if state["won"]:
        print(f"Victory! Final ante: {state['ante_num']}")
    else:
        print(f"Game over at ante {state['ante_num']}, round {state['round_num']}")

    return state["won"]


if __name__ == "__main__":
    play_game()