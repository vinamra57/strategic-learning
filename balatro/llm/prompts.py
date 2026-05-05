"""Prompt builders for the shop advisor.

Design rule (set by the user): the LLM is the strategy brain. Python's
job is to (a) pass through what the game returns, (b) describe the
mechanics — what each action does, what the inputs mean — and (c)
enforce legality. Python does NOT score items, categorise them, rank
them, give tier lists, give worked examples, or recommend strategies.

Every claim in the system prompt is:
  - a fact about how the game's mechanics work, OR
  - a fact about the action space / output schema, OR
  - a fact about the bot's behaviour (RL agent plays poker greedily;
    has no strategic awareness of jokers / discards / consumables —
    this is a constraint of THIS deployment, not a strategy claim).

No tier lists. No "use Mercury for pair-heavy". No worked examples of
correct picks. No "skip when shop is bad". The LLM derives strategy
from the game data it's given.
"""
from __future__ import annotations
from .state import ShopContext


# ─── System prompt: mechanics, action space, schema. No strategy. ────────
#
# Stylistic choices (deliberate):
#   • Game framing first so the LLM knows what world it's in.
#   • Action space described mechanically — what each action DOES, not
#     when to use it.
#   • Bot behaviour described factually — telling the LLM "the agent
#     plays greedily" isn't strategy advice, it's situational truth.
#   • Loop semantics explicit — multi-buy via repeated calls, skip = leave.
#   • Legality block lists what makes a response illegal so the LLM can
#     self-check before answering.
#   • User-prompt sections enumerated so the LLM knows where to look.
#   • Reasoning paragraph encourages careful thought without injecting
#     strategy. ("Read effect text" is process advice, not a tier list.)
#   • Output schema repeated in plain JSON so the LLM has a concrete
#     target. The format= schema parameter enforces it server-side.
#   • /no_think disables Qwen3's chain-of-thought (we picked the
#     -instruct variant which is post-trained without CoT).
SHOP_MAIN_SYSTEM = """\
You are the strategy brain for a Balatro bot. Balatro is a roguelike
poker deckbuilder: the bot plays poker hands against escalating chip
targets called "blinds". Between blinds the bot visits the shop —
that is where you act. You decide what to buy, what to open, when to
reroll, and when to leave.

A separate reinforcement-learning agent plays the poker hands. It
plays greedily: it picks the highest-scoring 5-card combo it can find
in its current hand, every time. It does NOT activate jokers, use
consumables (planet/tarot/spectral cards), modify cards in its deck,
or strategically time its discards. Effects that need the agent to
take a specific action to pay off won't pay off; effects that fire
automatically when a hand is played will work.

TWO MODES — the user prompt's first line tells you which one:

  ▸ SHOP MODE (header: "=== SHOP STATE ==="). The full shop is shown
    and you pick one of:
      buy_shop    — buy item at index N from "Shop items"
      buy_voucher — buy voucher at index N from "Vouchers"
      buy_pack    — open booster pack at index N from "Booster packs"
                    (after this, the next call will be in PACK MODE
                    so you can pick a card from inside)
      reroll      — pay reroll_cost (shown in user prompt) for new
                    shop items
      skip        — leave the shop, start the next blind

  ▸ PACK MODE (header: "=== PACK OPENED: ..."). A booster pack has
    been opened and you must pick one of its cards or skip. We REUSE
    the same JSON schema as SHOP MODE but only two action values are
    valid:
      buy_shop with index N → pick card N from this pack
      skip     with index null → skip the pack (waste it)
    In PACK MODE, do NOT use buy_voucher / buy_pack / reroll —
    those values are invalid here and will be rejected.
    Picking from a pack USES the card immediately (planets level
    their hand, tarots/spectrals apply their effect, jokers go into
    a joker slot).

LOOP STRUCTURE: each call is one step in an iterative loop. After
your action runs, the new state is fetched and you are asked again.
You can chain as many buys, opens, picks, and rerolls as you want in
one shop visit. In SHOP MODE, emit "skip" only when you genuinely
want to leave the shop. In PACK MODE, emit "skip" only when nothing
in the pack is worth picking.

GAME MECHANICS YOU WILL NOT FIND IN ITEM EFFECT TEXT (these are how the
game works; not all of them are spelled out in any item's description):

  • Vouchers are PERMANENT for the rest of the run. Once purchased,
    a voucher's effect applies for every future round of the run; it
    never expires, never has to be reactivated, and you cannot lose it.
    The shop voucher section refreshes once per ante.
  • Jokers, once purchased, occupy a joker slot and fire automatically
    every time the agent plays a hand, for the rest of the run, until
    you sell them.
  • Planet cards (purchased from shop or picked from a Celestial Pack)
    are CONSUMED on use: each one permanently levels up the named
    poker-hand type by a fixed amount of chips/mult. Once levelled,
    the boost applies to every future play of that hand for the rest
    of the run.
  • Tarot cards are CONSUMED on use and apply a one-time effect — but
    that effect (e.g. modifying a card in the deck) often persists
    permanently on the deck. Read the effect text.
  • Spectral cards are CONSUMED on use; effects vary widely.
  • Money interest: at the end of each round, the bot earns $1 in
    interest per $5 it currently holds, capped at $5/round (reached
    when holding $25 or more). Spending below $25 reduces this
    passive income.
  • Editions: items may have an edition tag in [brackets] — foil,
    holographic, polychrome, or negative. Editions add bonuses on top
    of the base effect text. Negative-edition jokers in particular
    do NOT consume a joker slot, so they bypass the joker-slot limit.
  • Joker duplicates: buying a joker label you already own gives you a
    second independent copy in another slot — it does not enhance the
    existing one. Each copy fires independently.
  • Pack contents: when you pick a card from a pack, it is USED
    immediately — planet cards level up the matched poker hand right
    away; tarots/spectrals apply their effect right away; jokers go
    into a joker slot. So opening a pack does NOT require a free
    consumable slot, only the slot needed for the specific picked card.
  • Reroll availability: reroll is always usable when you can afford
    its cost, even if every shop section is empty. An empty shop is
    not "done" — rerolling produces fresh items.
  • Reroll cost escalation: each reroll within a single shop visit
    increases the cost (typically by $1). The current value is shown
    in the user prompt; trust that number.

INPUTS you will see in the user prompt:
  - Money, ante, round, current reroll cost, slot capacities
  - The next blind (name, chip target, boss-special effect if any)
  - Owned jokers and their effect text
  - Owned consumables (planet/tarot/spectral cards stored to use later)
  - Active vouchers (already purchased; effects apply for the run)
  - Hand-play history: which poker-hand types the agent has actually
    played, with counts and average score per type
  - Shop items, shop vouchers, shop booster packs — each shown with
    index, label, set/category, edition (if any), cost in $, and the
    game's own effect description

LEGALITY (illegal actions are rejected; you'll retry with the reason):
  - the chosen index must be in range for the relevant section
  - the item's cost must be ≤ your current money
  - cannot buy a joker if joker slots are at max capacity
  - cannot reroll if money < reroll_cost
  - sections shown "(none in shop this round)" or "(empty)" have no
    valid index — do not target them

REASONING — fill the "scratchpad" field BEFORE the action. Be concise.

  STEP 1 — AFFORDABILITY. For each existing item, compute the comparison
  on one line. Get the math right by writing it explicitly:
      shop[i] LABEL: $cost <= $money? YES/NO
      voucher[i] LABEL: $cost <= $money? YES/NO
      pack[i] LABEL: $cost <= $money? YES/NO
      reroll: $reroll_cost <= $money? YES/NO
  Skip lines for empty sections.

  STEP 2 — LEGALITY. From the YES list: drop any Joker if joker slots
  are full (used == max). What remains is your candidate set.

  STEP 3 — VALUE. For each candidate, write ONE short line covering:
      - Quote a key phrase from its effect text verbatim. Do NOT add
        meanings the text does not contain. If the effect text uses
        terminology you cannot decode (e.g. an unfamiliar joker name
        with a cryptic effect), say so plainly — do not invent
        semantics or paraphrase what isn't there.
      - Duration class:
          PERMANENT-FOR-RUN  (voucher, planet level-up, owned-joker
                              firing every hand from now on)
          ONE-SHOT           (tarot, spectral, single-use effects)
          PER-ROUND          (effect that resets every round)
      - Trigger frequency given the agent's hand-play history (use
        the actual counts; unfamiliar trigger conditions = unknown
        frequency — do not assume).
      - Edition bonus (foil/holo/poly/negative) and owned-duplicate
        flag, if applicable.

  STEP 4 — DECIDE. Pick the highest-value candidate. If candidates
  are all weak and reroll is legal, reroll. If you're satisfied for
  this shop, skip.

OUTPUT — strict JSON, no prose, no markdown, no code fences:
{
  "scratchpad": "<your reasoning trace following the 4 steps above>",
  "action": "buy_shop" | "buy_voucher" | "buy_pack" | "reroll" | "skip",
  "index": <integer ≥0 or null>,
  "reason": "<one short sentence summary>"
}

Rules:
  - Fill scratchpad FIRST; then action/index/reason consistent with it.
  - "index" is required (non-negative integer) when action is one of
    buy_shop / buy_voucher / buy_pack.
  - "index" must be null when action is reroll or skip.
  - Output only the JSON object. No additional text.

/no_think
"""

# Same system prompt for pack-pick mode. The user prompt distinguishes
# the two modes. Sharing one system prefix lets Ollama keep its KV cache
# warm across calls of either kind (saves 5-7s per kind switch).
PACK_PICK_SYSTEM = SHOP_MAIN_SYSTEM


# ─── User-prompt rendering: raw passthrough of game state ─────────────────

def _fmt_history(ctx: ShopContext) -> str:
    h = ctx.history
    if h.total_plays == 0:
        return "  (no hands played yet — first shop of the run)"
    top = h.top_hands(9)
    avgs = h.avg_score_by_hand()
    lines = [
        f"  - {hand}: {count}× (avg score {avgs.get(hand, 0):.0f})"
        for hand, count in top
    ]
    return "\n".join(lines)


def _fmt_jokers(ctx: ShopContext) -> str:
    if not ctx.owned_jokers:
        return "  (none)"
    return "\n".join(
        f"  [{j.index}] {j.label}{f' ({j.edition})' if j.edition else ''} — "
        f"{j.effect}"
        for j in ctx.owned_jokers
    )


def _fmt_consumables(ctx: ShopContext) -> str:
    if not ctx.owned_consumables:
        return "  (none)"
    return "\n".join(
        f"  [{c.index}] {c.label} ({c.set}) — {c.effect}"
        for c in ctx.owned_consumables
    )


def _fmt_shop_items(ctx: ShopContext) -> str:
    if not ctx.shop_items:
        return "  (empty)"
    lines = []
    for it in ctx.shop_items:
        ed = f" [{it.edition}]" if it.edition else ""
        lines.append(
            f"  [{it.index}] {it.label}{ed} ({it.set}) — ${it.cost}\n"
            f"        {it.effect}"
        )
    return "\n".join(lines)


def _fmt_vouchers(ctx: ShopContext) -> str:
    if not ctx.vouchers:
        return "  (none in shop this round)"
    return "\n".join(
        f"  [{v.index}] {v.label} — ${v.cost}\n        {v.effect}"
        for v in ctx.vouchers
    )


def _fmt_packs(ctx: ShopContext) -> str:
    if not ctx.packs:
        return "  (none in shop this round)"
    return "\n".join(
        f"  [{p.index}] {p.label} — ${p.cost}\n        {p.effect}"
        for p in ctx.packs
    )


def _fmt_next_blind(ctx: ShopContext) -> str:
    if not ctx.next_blind_name:
        return "  (unknown)"
    line = f"  {ctx.next_blind_name}: {ctx.next_blind_chips} chips required"
    if ctx.next_blind_effect:
        line += f"\n  Effect: {ctx.next_blind_effect}"
    return line


def build_shop_user_prompt(ctx: ShopContext) -> str:
    used = ", ".join(ctx.used_vouchers) if ctx.used_vouchers else "(none)"
    parts = [
        "=== SHOP STATE ===",
        "",
        f"Ante {ctx.ante} / Round {ctx.round_num}",
        f"Money: ${ctx.money}",
        f"Reroll cost: ${ctx.reroll_cost}",
        # Spell out joker / consumable slot occupancy unambiguously.
        # Earlier scratchpads sometimes parsed "1/5 (4 free)" as "only
        # 1 slot available", so we render the three numbers separately.
        f"Joker slots: {ctx.joker_slots_used} occupied, "
        f"{ctx.joker_slots_max} max, {ctx.joker_slots_free} free",
        f"Consumable slots: {ctx.consumable_slots_used} occupied, "
        f"{ctx.consumable_slots_max} max, {ctx.consumable_slots_free} free",
        "",
        "Next blind:",
        _fmt_next_blind(ctx),
        "",
        "Owned jokers:",
        _fmt_jokers(ctx),
        "",
        "Owned consumables:",
        _fmt_consumables(ctx),
        "",
        f"Active vouchers: {used}",
        "",
        "Agent hand-play history:",
        _fmt_history(ctx),
        f"Total hands played: {ctx.history.total_plays}, "
        f"discards used: {ctx.history.discards_used}",
        "",
        "Shop items:",
        _fmt_shop_items(ctx),
        "",
        "Vouchers:",
        _fmt_vouchers(ctx),
        "",
        "Booster packs:",
        _fmt_packs(ctx),
        "",
        "Pick ONE action. Respond with strict JSON only.",
    ]
    return "\n".join(parts)


def build_pack_user_prompt(
    pack_label: str,
    cards: list[dict],
    pick_limit: int,
    ctx: ShopContext,
) -> str:
    """Pack-pick mode reuses the shop schema:
       action="buy_shop" + index=N → pick card N from this pack
       action="skip"     + index=null → skip the pack
       Other actions are invalid in pack mode."""
    card_lines = []
    for i, c in enumerate(cards):
        label = c.get("label", c.get("key", "?"))
        cset = c.get("set", "")
        eff = (c.get("value", {}) or {}).get("effect", "")
        card_lines.append(f"  [{i}] {label} ({cset}) — {eff}")
    cards_block = "\n".join(card_lines) if card_lines else "  (empty)"

    parts = [
        f"=== PACK OPENED: {pack_label} ===",
        "(You are in PACK MODE — see system prompt for the action mapping.)",
        f"Pick ONE card by index from the list below (using action=\"buy_shop\","
        f" index=N), or skip the pack (action=\"skip\", index=null).",
        f"You may pick up to {pick_limit} card(s) total in this pack across"
        f" sequential calls, but emit ONE pick per call. Cards you don't"
        f" pick are discarded when the pack closes.",
        "",
        f"Ante {ctx.ante} / Money ${ctx.money}",
        f"Joker slots: {ctx.joker_slots_used} occupied, "
        f"{ctx.joker_slots_max} max, {ctx.joker_slots_free} free",
        f"Consumable slots: {ctx.consumable_slots_used} occupied, "
        f"{ctx.consumable_slots_max} max, {ctx.consumable_slots_free} free",
        "",
        "Owned jokers:",
        _fmt_jokers(ctx),
        "",
        "Active vouchers: "
        + (", ".join(ctx.used_vouchers) if ctx.used_vouchers else "(none)"),
        "",
        "Next blind:",
        _fmt_next_blind(ctx),
        "",
        "Agent hand-play history:",
        _fmt_history(ctx),
        f"Total hands played: {ctx.history.total_plays}",
        "",
        "Cards in pack:",
        cards_block,
        "",
        "Reminder: in PACK MODE the only valid action values are",
        '  "buy_shop" (= pick card N) or "skip" (= leave the pack).',
        "Other action values will be rejected.",
        "",
        "Respond with strict JSON only.",
    ]
    return "\n".join(parts)


# Strict JSON schema enforced by Ollama's `format` parameter. Same schema
# for shop and pack modes; pack reinterprets buy_shop as "pick from pack".
SHOP_MAIN_SCHEMA = {
    "type": "object",
    # Order matters: scratchpad declared first so the LLM emits its
    # reasoning before committing to action/index. Acts as JSON-native
    # chain-of-thought without using <think> tokens.
    "properties": {
        "scratchpad": {"type": "string"},
        "action": {
            "type": "string",
            "enum": ["buy_shop", "buy_voucher", "buy_pack", "reroll", "skip"],
        },
        "index": {"type": ["integer", "null"], "minimum": 0},
        "reason": {"type": "string"},
    },
    "required": ["scratchpad", "action", "index", "reason"],
    "additionalProperties": False,
}

PACK_PICK_SCHEMA = SHOP_MAIN_SCHEMA
