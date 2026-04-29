# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
# ]
# ///

"""
Manual CLI for interacting with a live Balatro game via the balatrobot JSON-RPC API.
Useful for debugging, exploring game state structure, and learning the API.

Usage:
    python manual.py
    uv run manual.py
"""

import json
import sys
import textwrap

import requests

URL = "http://127.0.0.1:12346"

# ---------------------------------------------------------------------------
# RPC
# ---------------------------------------------------------------------------

def rpc(method: str, params: dict | None = None) -> dict:
    payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1}
    try:
        resp = requests.post(URL, json=payload, timeout=5)
        data = resp.json()
    except requests.exceptions.ConnectionError:
        print(f"\n[error] Cannot connect to {URL}. Is Balatro running with the balatrobot mod?\n")
        sys.exit(1)
    if "error" in data:
        raise RuntimeError(data["error"]["message"])
    return data["result"]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED   = "\033[91m"


def fmt_card(card: dict, index: int | None = None, highlighted: bool = False) -> str:
    v = card.get("value", {})
    rank = v.get("rank", "?")
    suit = v.get("suit", "?")
    mod  = card.get("modifier", {})
    state = card.get("state", {})

    # extras = []
    # if mod.get("enhancement"):
    #     extras.append(mod["enhancement"])
    # if mod.get("edition"):
    #     extras.append(mod["edition"])
    # if mod.get("seal"):
    #     extras.append(f"{mod['seal']} seal")
    # if state.get("debuff"):
    #     extras.append("DEBUFF")

    # extra_str = f" [{','.join(extras)}]" if extras else ""
    hl = f"{YELLOW}►{RESET}" if highlighted else " "
    idx_str = f"{DIM}[{index}]{RESET} " if index is not None else ""
    return f"{hl}{idx_str}{rank}{suit}{RESET}" # {extra_str}


def fmt_cards_area(area: dict, label: str, highlight_indices: list[int] | None = None) -> None:
    cards = area.get("cards", [])
    count = area.get("count", len(cards))
    limit = area.get("limit", "?")
    print(f"  {BOLD}{label}{RESET} ({count}/{limit}):")
    if not cards:
        print(f"    {DIM}(empty){RESET}")
        return
    for i, card in enumerate(cards):
        hl = highlight_indices is not None and i in highlight_indices
        print(f"    {fmt_card(card, index=i, highlighted=hl)}")


def fmt_joker(card: dict, index: int) -> str:
    label = card.get("label", card.get("key", "?"))
    effect = card.get("value", {}).get("effect", "")
    mod = card.get("modifier", {})
    parts = [f"{DIM}[{index}]{RESET} {CYAN}{label}{RESET}"]
    if effect:
        parts.append(f"{DIM}— {textwrap.shorten(effect, 60)}{RESET}")
    if mod.get("edition"):
        parts.append(f"[{mod['edition']}]")
    return "  ".join(parts)


def print_separator(char: str = "─", width: int = 60) -> None:
    print(f"{DIM}{char * width}{RESET}")


def print_header(title: str) -> None:
    print()
    print_separator("━")
    print(f"  {BOLD}{title}{RESET}")
    print_separator("━")


def print_game_info(state: dict) -> None:
    ante    = state.get("ante_num", "?")
    round_n = state.get("round_num", "?")
    money   = state.get("money", "?")
    seed    = state.get("seed", "?")
    rnd     = state.get("round") or {}
    hands   = rnd.get("hands_left", "?")
    discs   = rnd.get("discards_left", "?")
    chips   = rnd.get("chips", 0)

    print(f"  Ante {BOLD}{ante}{RESET}  Round {BOLD}{round_n}{RESET}  "
          f"Money {GREEN}${money}{RESET}  Seed {DIM}{seed}{RESET}")

    if rnd:
        print(f"  Chips scored: {YELLOW}{chips}{RESET}  "
              f"Hands left: {hands}  Discards left: {discs}")

    # Current blind
    blinds = state.get("blinds") or {}
    for blind in blinds.values():
        if isinstance(blind, dict) and blind.get("status") == "CURRENT":
            score = blind.get("score", "?")
            name  = blind.get("name", blind.get("type", "?"))
            effect = blind.get("effect", "")
            tag   = blind.get("tag_name", "")
            bar_pct = min(1.0, chips / score) if isinstance(chips, int) and isinstance(score, int) else 0
            filled = int(bar_pct * 20)
            bar = f"{GREEN}{'█' * filled}{DIM}{'░' * (20 - filled)}{RESET}"
            print(f"  Blind: {BOLD}{name}{RESET}  Target: {RED}{score}{RESET}  [{bar}]")
            if effect and effect != "No special effect":
                print(f"  Effect: {DIM}{effect}{RESET}")
            if tag:
                skip_tag = blind.get("tag_effect", "")
                print(f"  Skip tag: {CYAN}{tag}{RESET}" + (f" — {skip_tag}" if skip_tag else ""))


def print_blinds(state: dict) -> None:
    blinds = state.get("blinds") or {}
    print(f"\n  {'Type':<10} {'Name':<20} {'Score':>8}  {'Status':<12}  Tag on skip")
    print_separator()
    for btype, blind in blinds.items():
        if not isinstance(blind, dict):
            continue
        status  = blind.get("status", "")
        name    = blind.get("name", btype)
        score   = blind.get("score", "")
        tag     = blind.get("tag_name", "")
        effect  = blind.get("effect", "")
        marker  = f"{YELLOW}◄ current{RESET}" if status == "CURRENT" else \
                  f"{GREEN}✓{RESET}" if status in ("DEFEATED", "SKIPPED") else \
                  f"{DIM}{status.lower()}{RESET}"
        print(f"  {btype:<10} {name:<20} {str(score):>8}  {marker:<20}  {tag}")
        if effect and effect != "No special effect":
            print(f"  {DIM}{'':>10} {effect}{RESET}")


def print_jokers(state: dict) -> None:
    jokers = state.get("jokers") or {}
    cards  = jokers.get("cards", [])
    if not cards:
        return
    print(f"\n  {BOLD}Jokers{RESET} ({jokers.get('count', 0)}/{jokers.get('limit', '?')}):")
    for i, j in enumerate(cards):
        print(f"    {fmt_joker(j, i)}")


def print_consumables(state: dict) -> None:
    cons  = state.get("consumables") or {}
    cards = cons.get("cards", [])
    if not cards:
        return
    print(f"\n  {BOLD}Consumables{RESET} ({cons.get('count', 0)}/{cons.get('limit', '?')}):")
    for i, c in enumerate(cards):
        label  = c.get("label", c.get("key", "?"))
        effect = c.get("value", {}).get("effect", "")
        print(f"    {DIM}[{i}]{RESET} {label}  {DIM}{textwrap.shorten(effect, 55)}{RESET}")


def print_shop(state: dict) -> None:
    shop     = state.get("shop") or {}
    vouchers = state.get("vouchers") or {}
    packs    = state.get("packs") or {}

    for label, area in [("Items", shop), ("Vouchers", vouchers), ("Packs", packs)]:
        cards = area.get("cards", [])
        if not cards:
            continue
        print(f"\n  {BOLD}{label}{RESET}:")
        for i, c in enumerate(cards):
            name   = c.get("label", c.get("key", "?"))
            effect = c.get("value", {}).get("effect", "")
            cost   = c.get("cost", {}).get("buy", "?")
            print(f"    {DIM}[{i}]{RESET} {name:<25} {GREEN}${cost}{RESET}  "
                  f"{DIM}{textwrap.shorten(effect, 40)}{RESET}")


def print_pack(state: dict) -> None:
    pack  = state.get("pack") or {}
    cards = pack.get("cards", [])
    count = pack.get("count", len(cards))
    limit = pack.get("limit", "?")
    print(f"\n  {BOLD}Pack contents{RESET} ({count} cards, pick up to {limit}):")
    for i, c in enumerate(cards):
        label  = c.get("label", c.get("key", "?"))
        effect = c.get("value", {}).get("effect", "")
        cset   = c.get("set", "")
        print(f"    {DIM}[{i}]{RESET} {label:<25} {DIM}[{cset}]{RESET}  "
              f"{DIM}{textwrap.shorten(effect, 40)}{RESET}")


def print_state(state: dict) -> None:
    game_state = state.get("state", "UNKNOWN")
    print_header(f"State: {CYAN}{game_state}{RESET}")
    print_game_info(state)
    print_jokers(state)
    print_consumables(state)

    match game_state:
        case "BLIND_SELECT":
            print_blinds(state)
        case "SELECTING_HAND":
            hand = state.get("hand") or {}
            print()
            fmt_cards_area(hand, "Hand")
        case "SHOP":
            print_shop(state)
        case "SMODS_BOOSTER_OPENED":
            print_pack(state)
        case "ROUND_EVAL":
            rnd = state.get("round") or {}
            print(f"\n  {BOLD}Round complete!{RESET}  Chips scored: {YELLOW}{rnd.get('chips', '?')}{RESET}")
        case "GAME_OVER":
            won = state.get("won", False)
            if won:
                print(f"\n  {GREEN}{BOLD}YOU WIN!{RESET}")
            else:
                print(f"\n  {RED}{BOLD}GAME OVER{RESET}")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def prompt(msg: str) -> str:
    try:
        return input(f"\n{BOLD}{msg}{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def pick_indices(cards: list, label: str, min_count: int = 1, max_count: int = 5) -> list[int]:
    """Prompt the user to pick card indices from a list."""
    print(f"\n  Enter space-separated indices ({min_count}–{max_count} cards):")
    for i, c in enumerate(cards):
        print(f"    {fmt_card(c, index=i)}")
    while True:
        raw = prompt(f"  Select {label} indices:")
        try:
            indices = [int(x) for x in raw.split()]
        except ValueError:
            print("  [!] Enter numbers separated by spaces.")
            continue
        if not (min_count <= len(indices) <= max_count):
            print(f"  [!] Pick between {min_count} and {max_count} cards.")
            continue
        if any(i < 0 or i >= len(cards) for i in indices):
            print(f"  [!] Indices must be 0–{len(cards) - 1}.")
            continue
        return indices


def show_raw(state: dict) -> None:
    print(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Action menus per game state
# ---------------------------------------------------------------------------

def menu_start() -> dict:
    """Present the start-game menu and return the initial state."""
    print_header("BalatroBot Manual CLI")

    decks  = ["RED","BLUE","YELLOW","GREEN","BLACK","MAGIC","NEBULA","GHOST",
              "ABANDONED","CHECKERED","ZODIAC","PAINTED","ANAGLYPH","PLASMA","ERRATIC"]
    stakes = ["WHITE","RED","GREEN","BLACK","BLUE","PURPLE","ORANGE","GOLD"]

    print(f"\n  {BOLD}Decks:{RESET}")
    for i, d in enumerate(decks):
        print(f"    [{i}] {d}")
    deck_idx = prompt("Deck [0=RED]:")
    deck = decks[int(deck_idx)] if deck_idx.isdigit() and int(deck_idx) < len(decks) else "RED"

    print(f"\n  {BOLD}Stakes:{RESET}")
    for i, s in enumerate(stakes):
        print(f"    [{i}] {s}")
    stake_idx = prompt("Stake [0=WHITE]:")
    stake = stakes[int(stake_idx)] if stake_idx.isdigit() and int(stake_idx) < len(stakes) else "WHITE"

    print(f"\n  Starting game: deck={deck} stake={stake} ...")
    init  = rpc("menu")
    state = rpc("start", {"deck": deck, "stake": stake})
    print(f"  Seed: {CYAN}{state.get('seed')}{RESET}")
    return state


def handle_blind_select(state: dict) -> dict:
    blinds = state.get("blinds") or {}
    # Find the SELECT blind (the one we're choosing on)
    selectable = {k: v for k, v in blinds.items()
                  if isinstance(v, dict) and v.get("status") == "SELECT"}

    print(f"\n  Options:")
    print(f"    [s] Select blind")
    if selectable:
        print(f"    [k] Skip blind")
    print(f"    [r] Show raw state")

    choice = prompt("Action:")
    match choice.lower():
        case "s":
            return rpc("select")
        case "k" if selectable:
            try:
                return rpc("skip")
            except RuntimeError as e:
                print(f"  [!] Skip failed: {e}")
                return state
        case "r":
            show_raw(state)
            return state
        case _:
            print("  [!] Unknown option.")
            return state


def handle_selecting_hand(state: dict) -> dict:
    hand  = state.get("hand") or {}
    cards = hand.get("cards", [])
    rnd   = state.get("round") or {}
    discards_left = rnd.get("discards_left", 0)

    print(f"\n  Options:")
    print(f"    [p] Play cards  (choose up to 5)")
    if discards_left > 0:
        print(f"    [d] Discard cards  ({discards_left} discards left)")
    print(f"    [r] Show raw state")

    choice = prompt("Action:")
    match choice.lower():
        case "p":
            indices = pick_indices(cards, "cards to PLAY", min_count=1, max_count=5)
            print(f"  Playing: {[fmt_card(cards[i]) for i in indices]}")
            return rpc("play", {"cards": indices})
        case "d" if discards_left > 0:
            indices = pick_indices(cards, "cards to DISCARD", min_count=1, max_count=5)
            print(f"  Discarding: {[fmt_card(cards[i]) for i in indices]}")
            return rpc("discard", {"cards": indices})
        case "r":
            show_raw(state)
            return state
        case _:
            print("  [!] Unknown option.")
            return state


def handle_round_eval(state: dict) -> dict:
    print(f"\n  Options:")
    print(f"    [c] Cash out")
    print(f"    [r] Show raw state")

    choice = prompt("Action:")
    match choice.lower():
        case "c":
            return rpc("cash_out")
        case "r":
            show_raw(state)
            return state
        case _:
            print("  [!] Unknown option.")
            return state


def handle_shop(state: dict) -> dict:
    money    = state.get("money", 0)
    shop     = state.get("shop") or {}
    vouchers = state.get("vouchers") or {}
    packs    = state.get("packs") or {}

    shop_cards    = shop.get("cards", [])
    voucher_cards = vouchers.get("cards", [])
    pack_cards    = packs.get("cards", [])

    print(f"\n  Options:")
    print(f"    [n] Next round  (leave shop)")
    if shop_cards:
        print(f"    [b] Buy shop item  (0–{len(shop_cards)-1})")
    if voucher_cards:
        print(f"    [v] Buy voucher    (0–{len(voucher_cards)-1})")
    if pack_cards:
        print(f"    [p] Buy pack       (0–{len(pack_cards)-1})")
    print(f"    [r] Show raw state")

    choice = prompt("Action:")
    match choice.lower():
        case "n":
            return rpc("next_round")
        case "b" if shop_cards:
            idx = prompt(f"  Shop item index (0–{len(shop_cards)-1}):")
            if idx.isdigit() and int(idx) < len(shop_cards):
                try:
                    return rpc("buy", {"card": int(idx)})
                except RuntimeError as e:
                    print(f"  [!] Buy failed: {e}")
            return state
        case "v" if voucher_cards:
            idx = prompt(f"  Voucher index (0–{len(voucher_cards)-1}):")
            if idx.isdigit() and int(idx) < len(voucher_cards):
                try:
                    return rpc("buy", {"card": int(idx)})
                except RuntimeError as e:
                    print(f"  [!] Buy failed: {e}")
            return state
        case "p" if pack_cards:
            idx = prompt(f"  Pack index (0–{len(pack_cards)-1}):")
            if idx.isdigit() and int(idx) < len(pack_cards):
                try:
                    return rpc("buy", {"pack": int(idx)})
                except RuntimeError as e:
                    print(f"  [!] Buy failed: {e}")
            return state
        case "r":
            show_raw(state)
            return state
        case _:
            print("  [!] Unknown option.")
            return state


def handle_pack(state: dict) -> dict:
    pack  = state.get("pack") or {}
    cards = pack.get("cards", [])

    print(f"\n  Options:")
    if cards:
        print(f"    [c] Pick a card  (0–{len(cards)-1})")
    print(f"    [s] Skip pack")
    print(f"    [r] Show raw state")

    choice = prompt("Action:")
    match choice.lower():
        case "c" if cards:
            idx = prompt(f"  Card index (0–{len(cards)-1}):")
            if idx.isdigit() and int(idx) < len(cards):
                try:
                    return rpc("pack", {"card": int(idx)})
                except RuntimeError as e:
                    print(f"  [!] Pick failed: {e}")
            return state
        case "s":
            try:
                result = rpc("pack", {"skip": True})
                return rpc("gamestate")  # flush stale pack data
            except RuntimeError as e:
                print(f"  [!] Skip failed: {e}")
                return state
        case "r":
            show_raw(state)
            return state
        case _:
            print("  [!] Unknown option.")
            return state


# ---------------------------------------------------------------------------
# Global commands available from any state
# ---------------------------------------------------------------------------

GLOBAL_HELP = f"""
  {BOLD}Global commands (available at any prompt):{RESET}
    raw       — print full raw JSON of current state
    gs        — re-fetch gamestate from server
    quit / q  — exit
"""


def handle_global(choice: str, state: dict) -> dict | None:
    """Returns new state if handled, None if not a global command."""
    match choice.lower():
        case "raw":
            show_raw(state)
            return state
        case "gs":
            return rpc("gamestate")
        case "quit" | "q":
            print("Bye.")
            sys.exit(0)
        case "help" | "?":
            print(GLOBAL_HELP)
            return state
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    state = menu_start()

    while True:
        print_state(state)
        game_state = state.get("state", "UNKNOWN")

        if game_state == "GAME_OVER":
            print()
            again = prompt("Play again? [y/n]:")
            if again.lower() == "y":
                state = menu_start()
            else:
                print("Bye.")
                break
            continue

        # Dispatch to the state handler. Each handler returns the new state
        # (possibly unchanged if the player just viewed raw output).
        match game_state:
            case "BLIND_SELECT":
                new = handle_blind_select(state)
            case "SELECTING_HAND":
                new = handle_selecting_hand(state)
            case "ROUND_EVAL":
                new = handle_round_eval(state)
            case "SHOP":
                new = handle_shop(state)
            case "SMODS_BOOSTER_OPENED":
                new = handle_pack(state)
            case _:
                print(f"  {DIM}Unknown state '{game_state}' — fetching gamestate...{RESET}")
                new = rpc("gamestate")

        # Check if the input was actually a global command
        if new is state:
            pass  # state unchanged, re-display on next iteration
        else:
            state = new


if __name__ == "__main__":
    main()
