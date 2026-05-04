"""Top-level shop advisor.

Every decision goes to the LLM. Python only handles:
  - rendering raw game state into the user prompt
  - validating the LLM's choice for legality (cost ≤ money, in-range
    indices, joker-slot capacity, reroll buffer)
  - retrying with corrective feedback when the LLM picks something illegal
  - logging
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

from .actions import (
    BuyShopItem, BuyVoucher, BuyPack, Reroll, Skip, ShopAction, PackPick,
)
from .prompts import (
    SHOP_MAIN_SYSTEM,
    PACK_PICK_SYSTEM,
    SHOP_MAIN_SCHEMA,
    PACK_PICK_SCHEMA,
    build_shop_user_prompt,
    build_pack_user_prompt,
)
from .runner import OllamaRunner
from .state import ShopContext


class ShopAdvisor:
    def __init__(
        self,
        model: str = "qwen3:4b-instruct",
        log_path: Optional[str] = "logs/shop_advisor.jsonl",
        url: str = "http://localhost:11434",
        timeout: float = 30.0,
        warmup: bool = True,
    ):
        self.runner = OllamaRunner(model=model, url=url, timeout=timeout)
        self.log_path = Path(log_path) if log_path else None
        self.llm_calls = 0
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if warmup and self.runner.health():
            try:
                self.runner.chat(
                    system=SHOP_MAIN_SYSTEM,
                    user="warmup — return action skip with index null.",
                    schema=SHOP_MAIN_SCHEMA,
                )
            except Exception:
                pass

    # ─── decision API ────────────────────────────────────────────────

    def decide_shop(self, ctx: ShopContext, max_corrections: int = 2) -> ShopAction:
        user = build_shop_user_prompt(ctx)
        correction_note = ""

        for attempt in range(max_corrections + 1):
            full_user = user + correction_note
            self.llm_calls += 1
            try:
                parsed = self.runner.chat(
                    SHOP_MAIN_SYSTEM, full_user, schema=SHOP_MAIN_SCHEMA,
                )
            except RuntimeError as e:
                self._log("shop", ctx, None, error=str(e))
                return Skip(reason=f"advisor error: {e!r}")

            action, illegal_reason = self._coerce_shop(parsed, ctx)
            if illegal_reason is None:
                self._log("shop", ctx, parsed, action=action)
                return action

            if attempt < max_corrections:
                correction_note = (
                    f"\n\nYour previous response was rejected: {illegal_reason}. "
                    "Pick a legal action."
                )
                continue

            self._log("shop", ctx, parsed, action=action,
                      extra={"degraded": True, "reason": illegal_reason})
            return action

        return Skip(reason="no decision reached")

    def decide_pack(
        self,
        pack_label: str,
        cards: list[dict],
        pick_limit: int,
        ctx: ShopContext,
        max_corrections: int = 2,
    ) -> PackPick:
        user = build_pack_user_prompt(pack_label, cards, pick_limit, ctx)
        correction_note = ""

        for attempt in range(max_corrections + 1):
            full_user = user + correction_note
            self.llm_calls += 1
            try:
                parsed = self.runner.chat(
                    PACK_PICK_SYSTEM, full_user, schema=PACK_PICK_SCHEMA,
                )
            except RuntimeError as e:
                self._log("pack", ctx, None, error=str(e),
                          extra={"pack": pack_label})
                return PackPick(index=None, reason=f"advisor error: {e!r}")

            pick, illegal_reason = self._coerce_pack(parsed, cards)
            if illegal_reason is None:
                self._log("pack", ctx, parsed, action=pick,
                          extra={"pack": pack_label})
                return pick

            if attempt < max_corrections:
                correction_note = (
                    f"\n\nYour previous response was rejected: {illegal_reason}. "
                    "Pick a legal action."
                )
                continue

            self._log("pack", ctx, parsed, action=pick,
                      extra={"pack": pack_label, "degraded": True,
                             "reason": illegal_reason})
            return pick

        return PackPick(index=None, reason="no decision reached")

    # ─── coercion: parsed JSON → typed action, with legality clamps ──

    def _coerce_shop(
        self, parsed: dict, ctx: ShopContext,
    ) -> tuple[ShopAction, str | None]:
        action = parsed.get("action")
        idx = parsed.get("index")
        reason = parsed.get("reason", "")

        if action == "buy_shop":
            if not isinstance(idx, int) or idx < 0 or idx >= len(ctx.shop_items):
                return Skip(reason=f"illegal shop index {idx}"), \
                    f"buy_shop index {idx} out of range 0..{len(ctx.shop_items)-1}"
            item = ctx.shop_items[idx]
            if item.cost > ctx.money:
                return Skip(reason=f"can't afford {item.label}"), \
                    f"shop[{idx}] cost ${item.cost} > money ${ctx.money}"
            if item.set == "Joker" and ctx.joker_slots_free <= 0:
                return Skip(reason="joker slots full"), \
                    f"shop[{idx}] is a Joker but joker slots are full"
            return BuyShopItem(index=idx, reason=reason), None

        if action == "buy_voucher":
            if not isinstance(idx, int) or idx < 0 or idx >= len(ctx.vouchers):
                if not ctx.vouchers:
                    return Skip(reason="no vouchers in shop"), \
                        "buy_voucher chosen but no vouchers in shop this round"
                return Skip(reason=f"illegal voucher index {idx}"), \
                    f"buy_voucher index {idx} out of range 0..{len(ctx.vouchers)-1}"
            v = ctx.vouchers[idx]
            if v.cost > ctx.money:
                return Skip(reason="can't afford voucher"), \
                    f"voucher[{idx}] cost ${v.cost} > money ${ctx.money}"
            return BuyVoucher(index=idx, reason=reason), None

        if action == "buy_pack":
            if not isinstance(idx, int) or idx < 0 or idx >= len(ctx.packs):
                if not ctx.packs:
                    return Skip(reason="no packs in shop"), \
                        "buy_pack chosen but no packs in shop this round"
                return Skip(reason=f"illegal pack index {idx}"), \
                    f"buy_pack index {idx} out of range 0..{len(ctx.packs)-1}"
            p = ctx.packs[idx]
            if p.cost > ctx.money:
                return Skip(reason="can't afford pack"), \
                    f"pack[{idx}] cost ${p.cost} > money ${ctx.money}"
            return BuyPack(index=idx, reason=reason), None

        if action == "reroll":
            if ctx.money < ctx.reroll_cost:
                return Skip(reason="can't afford reroll"), \
                    f"reroll cost ${ctx.reroll_cost} > money ${ctx.money}"
            return Reroll(reason=reason), None

        if action == "skip":
            return Skip(reason=reason), None

        return Skip(reason=f"unknown action {action!r}"), \
            f"unknown action {action!r}"

    def _coerce_pack(
        self, parsed: dict, cards: list[dict],
    ) -> tuple[PackPick, str | None]:
        action = parsed.get("action")
        idx = parsed.get("index")
        reason = parsed.get("reason", "")

        # In pack mode, "buy_shop" means pick that card; "skip" means skip pack.
        if action == "buy_shop":
            if not isinstance(idx, int) or idx < 0 or idx >= len(cards):
                return PackPick(index=None, reason="illegal pack index"), \
                    f"pack pick index {idx} out of range 0..{len(cards)-1}"
            return PackPick(index=idx, reason=reason), None

        if action == "skip":
            return PackPick(index=None, reason=reason or "skipped"), None

        # Other actions are invalid in pack mode
        return PackPick(index=None, reason=f"invalid pack action {action!r}"), \
            f"action {action!r} is invalid in pack mode (use buy_shop to pick or skip)"

    # ─── logging ─────────────────────────────────────────────────────

    def _log(
        self,
        kind: str,
        ctx: ShopContext,
        parsed: dict | None,
        action: ShopAction | PackPick | None = None,
        error: str | None = None,
        extra: dict | None = None,
    ) -> None:
        if self.log_path is None:
            return

        stats = self.runner.last_stats
        entry = {
            "ts": time.time(),
            "kind": kind,
            "model": self.runner.model,
            "ctx": {
                "money": ctx.money,
                "ante": ctx.ante,
                "round": ctx.round_num,
                "reroll_cost": ctx.reroll_cost,
                "joker_slots": [ctx.joker_slots_used, ctx.joker_slots_max],
                "shop_items": [
                    {"i": s.index, "label": s.label, "set": s.set, "cost": s.cost}
                    for s in ctx.shop_items
                ],
                "vouchers": [
                    {"i": v.index, "label": v.label, "cost": v.cost}
                    for v in ctx.vouchers
                ],
                "packs": [
                    {"i": p.index, "label": p.label, "cost": p.cost}
                    for p in ctx.packs
                ],
                "owned_jokers": [j.label for j in ctx.owned_jokers],
                "history_top": ctx.history.top_hands(5),
                "history_total": ctx.history.total_plays,
            },
            "parsed": parsed,
            "action": _action_dict(action) if action else None,
            "latency_s": stats.latency_s if stats else None,
            "error": error,
        }
        if extra:
            entry["extra"] = extra
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def _action_dict(a) -> dict:
    if a is None:
        return {}
    return {"kind": a.kind, **{k: v for k, v in a.__dict__.items()}}
