"""Typed state structures fed to the advisor.

These mirror balatrobot's GameState/Area/Card schema (see
https://github.com/coder/balatrobot OpenRPC spec) but flatten everything
the advisor actually needs into plain dataclasses, so we don't ship raw
nested dicts into the prompt builder.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class ShopItem:
    index: int
    label: str          # e.g. "Joker", "Pluto", "The Tower"
    set: str            # "Joker" | "Planet" | "Tarot" | "Spectral" | "Default" (playing card)
    effect: str         # description text from value.effect
    cost: int           # buy cost from cost.buy
    edition: str = ""   # "" | "foil" | "holographic" | "polychrome" | "negative" (from modifier.edition)


@dataclass(frozen=True)
class VoucherItem:
    index: int
    label: str
    effect: str
    cost: int


@dataclass(frozen=True)
class PackItem:
    index: int
    label: str          # e.g. "Celestial Pack", "Buffoon Pack", "Mega Arcana Pack"
    effect: str
    cost: int
    set: str = ""       # what kind of cards it contains (rough hint)


@dataclass(frozen=True)
class OwnedJoker:
    index: int
    label: str
    effect: str
    edition: str = ""
    sell_value: int = 0


@dataclass(frozen=True)
class OwnedConsumable:
    index: int
    label: str
    set: str            # "Planet" | "Tarot" | "Spectral"
    effect: str


# ─── Hand-play history ────────────────────────────────────────────────────

@dataclass
class HandHistory:
    """
    Tracks what hand types the RL agent has been playing this run.
    Updated by test_agent.py on every PLAY action.

    The advisor uses this to decide whether to upgrade specific poker
    hands (planet cards) or pick conditional jokers (e.g. +mult on Pair).
    """
    plays: list[tuple[str, int]] = field(default_factory=list)   # (hand_type, score)
    discards_used: int = 0

    def record_play(self, hand_type: str, score: int) -> None:
        self.plays.append((hand_type, score))

    def record_discard(self) -> None:
        self.discards_used += 1

    @property
    def total_plays(self) -> int:
        return len(self.plays)

    def hand_type_distribution(self, last_n: int | None = None) -> dict[str, int]:
        sample = self.plays[-last_n:] if last_n else self.plays
        return dict(Counter(h for h, _ in sample))

    def top_hands(self, k: int = 3) -> list[tuple[str, int]]:
        """Return [(hand_type, count), ...] sorted by frequency, top-k."""
        dist = self.hand_type_distribution()
        return sorted(dist.items(), key=lambda kv: -kv[1])[:k]

    def avg_score_by_hand(self) -> dict[str, float]:
        bucket: dict[str, list[int]] = {}
        for h, s in self.plays:
            bucket.setdefault(h, []).append(s)
        return {h: sum(v) / len(v) for h, v in bucket.items()}


# ─── Shop context (the thing the advisor sees) ────────────────────────────

@dataclass
class ShopContext:
    money: int
    ante: int
    round_num: int
    reroll_cost: int
    joker_slots_used: int
    joker_slots_max: int
    consumable_slots_used: int
    consumable_slots_max: int

    # Next blind the agent will face after leaving the shop. Data the
    # game tells a player but worth surfacing explicitly so the LLM can
    # gauge how much scaling is needed.
    next_blind_name: str = ""
    next_blind_chips: int = 0
    next_blind_effect: str = ""        # boss blinds have special effects

    shop_items: list[ShopItem] = field(default_factory=list)
    vouchers: list[VoucherItem] = field(default_factory=list)
    packs: list[PackItem] = field(default_factory=list)

    owned_jokers: list[OwnedJoker] = field(default_factory=list)
    owned_consumables: list[OwnedConsumable] = field(default_factory=list)
    used_vouchers: list[str] = field(default_factory=list)

    history: HandHistory = field(default_factory=HandHistory)

    # ── derived helpers ──────────────────────────────────────────────────

    @property
    def joker_slots_free(self) -> int:
        return max(0, self.joker_slots_max - self.joker_slots_used)

    @property
    def consumable_slots_free(self) -> int:
        return max(0, self.consumable_slots_max - self.consumable_slots_used)

    def can_afford(self, cost: int) -> bool:
        return self.money >= cost

    def affordable_indices(self, items: Iterable) -> list[int]:
        """Indices in `items` whose cost ≤ money."""
        return [it.index for it in items if it.cost <= self.money]


# ─── Parser: balatrobot raw state → ShopContext ───────────────────────────

_CONSUMABLE_SETS = {"Planet", "Tarot", "Spectral"}


def _safe_get(d: dict, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _card_to_shop_item(c: dict, idx: int) -> ShopItem:
    return ShopItem(
        index=idx,
        label=c.get("label", c.get("key", "?")),
        set=c.get("set", ""),
        effect=_safe_get(c, "value", "effect", default=""),
        cost=_safe_get(c, "cost", "buy", default=999),
        edition=_safe_get(c, "modifier", "edition", default="") or "",
    )


def parse_shop_context(state: dict, history: HandHistory) -> ShopContext:
    """Convert a balatrobot SHOP-state dict into a ShopContext."""
    rnd = state.get("round") or {}
    jokers_area = state.get("jokers") or {}
    cons_area = state.get("consumables") or {}
    shop_area = state.get("shop") or {}
    voucher_area = state.get("vouchers") or {}
    pack_area = state.get("packs") or {}

    shop_items = [
        _card_to_shop_item(c, i) for i, c in enumerate(shop_area.get("cards", []))
    ]
    vouchers = [
        VoucherItem(
            index=i,
            label=c.get("label", c.get("key", "?")),
            effect=_safe_get(c, "value", "effect", default=""),
            cost=_safe_get(c, "cost", "buy", default=999),
        )
        for i, c in enumerate(voucher_area.get("cards", []))
    ]
    packs = [
        PackItem(
            index=i,
            label=c.get("label", c.get("key", "?")),
            effect=_safe_get(c, "value", "effect", default=""),
            cost=_safe_get(c, "cost", "buy", default=999),
            set=c.get("set", ""),
        )
        for i, c in enumerate(pack_area.get("cards", []))
    ]
    owned_jokers = [
        OwnedJoker(
            index=i,
            label=c.get("label", c.get("key", "?")),
            effect=_safe_get(c, "value", "effect", default=""),
            edition=_safe_get(c, "modifier", "edition", default="") or "",
            sell_value=_safe_get(c, "cost", "sell", default=0),
        )
        for i, c in enumerate(jokers_area.get("cards", []))
    ]
    owned_consumables = [
        OwnedConsumable(
            index=i,
            label=c.get("label", c.get("key", "?")),
            set=c.get("set", ""),
            effect=_safe_get(c, "value", "effect", default=""),
        )
        for i, c in enumerate(cons_area.get("cards", []))
    ]

    used = state.get("used_vouchers") or {}
    used_list = list(used.values()) if isinstance(used, dict) else []

    # Next blind the player will face. During SHOP, the just-defeated
    # blind shows DEFEATED/SKIPPED; the next is UPCOMING or SELECT.
    blinds = state.get("blinds") or {}
    nb_name = nb_effect = ""
    nb_chips = 0
    for b in blinds.values():
        if not isinstance(b, dict):
            continue
        status = b.get("status", "")
        if status in ("UPCOMING", "SELECT", "CURRENT"):
            nb_name = b.get("name") or b.get("type") or ""
            nb_chips = b.get("score", 0)
            eff = b.get("effect", "")
            if eff and eff != "No special effect":
                nb_effect = eff
            break

    return ShopContext(
        money=state.get("money", 0),
        ante=state.get("ante_num", 1),
        round_num=state.get("round_num", 1),
        reroll_cost=rnd.get("reroll_cost", 5),
        joker_slots_used=jokers_area.get("count", 0),
        joker_slots_max=jokers_area.get("limit", 5),
        consumable_slots_used=cons_area.get("count", 0),
        consumable_slots_max=cons_area.get("limit", 2),
        next_blind_name=nb_name,
        next_blind_chips=nb_chips,
        next_blind_effect=nb_effect,
        shop_items=shop_items,
        vouchers=vouchers,
        packs=packs,
        owned_jokers=owned_jokers,
        owned_consumables=owned_consumables,
        used_vouchers=used_list,
        history=history,
    )
