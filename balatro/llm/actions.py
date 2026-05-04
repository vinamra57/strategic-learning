"""Typed actions returned by the shop advisor."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Union


@dataclass(frozen=True)
class BuyShopItem:
    index: int
    reason: str = ""
    kind: Literal["shop"] = "shop"


@dataclass(frozen=True)
class BuyVoucher:
    index: int
    reason: str = ""
    kind: Literal["voucher"] = "voucher"


@dataclass(frozen=True)
class BuyPack:
    index: int
    reason: str = ""
    kind: Literal["pack"] = "pack"


@dataclass(frozen=True)
class Reroll:
    reason: str = ""
    kind: Literal["reroll"] = "reroll"


@dataclass(frozen=True)
class Skip:
    """Leave the shop, advance to next round."""
    reason: str = ""
    kind: Literal["skip"] = "skip"


@dataclass(frozen=True)
class PackPick:
    """Pick a card from an open booster pack, or skip the pack."""
    index: int | None  # None = skip pack
    reason: str = ""
    kind: Literal["pack_pick"] = "pack_pick"


ShopAction = Union[BuyShopItem, BuyVoucher, BuyPack, Reroll, Skip]
