# strategic-learning

Reinforcement-learning agents for two card games, paired with a small
local-LLM strategy module for the meta-decisions RL alone doesn't cover.

## Subprojects

### `blackjack/`
Tabular Q-learning, policy-gradient, and actor-critic agents for
blackjack, plus a Hi-Lo card-counting LLM advisor (`llm_advisor.py` at
the repo root, also used at evaluation time).

### `balatro/`
PPO agent that plays single Balatro rounds (8-card hand, 4 plays,
4 discards, fixed chip target — see `balatro/ppoagent.py`), and an
LLM-driven advisor (`balatro/llm/`) for every meta-decision the RL
agent isn't trained for: which shop items / vouchers / packs to buy,
when to reroll, when to leave the shop, and which card to pick from
an opened booster pack. The integrated runner is `balatro/test_llm_agent.py`
(`balatro/test_agent.py` remains the PPO-only baseline runner).

The advisor speaks JSON-RPC to the
[balatrobot](https://github.com/coder/balatrobot) mod, so the same
runner drives a live game.

## Quick start

```bash
# 1. Environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python stable-baselines3 gymnasium numpy requests

# 2. Local LLM (shop advisor)
brew install ollama
ollama serve &
ollama pull qwen3:4b-instruct

# 3. Run a live game (Balatro launched with the balatrobot mod, listening on :12346):
python balatro/test_llm_agent.py --games 1
```

## Repo layout

```
.
├── blackjack/                       Blackjack RL stack
│   ├── model.py                     Tabular Q + PG + AC agents
│   ├── sim.py / optimal.py          Game sim + basic-strategy reference
│   └── benchmark.py                 Win-rate eval
├── balatro/                         Balatro RL + shop brain
│   ├── ppoagent.py                  Single-round env + PPO training
│   ├── test_agent.py                PPO-only baseline runner
│   ├── test_llm_agent.py                Live runner — PPO plays hands, LLM runs shop
│   └── llm/                         Shop / pack / reroll advisor (Ollama)
├── llm_advisor.py                   Hi-Lo LLM advisor for blackjack
└── precompute.py                    Precompute LLM lookup table for blackjack
```

## Design notes — Balatro shop advisor

`balatro/llm/` is intentionally minimal: the LLM is the strategist, and
Python is only responsible for (a) rendering raw game state into a
single user prompt, (b) validating the model's chosen action against
game-legality rules (cost ≤ money, in-range index, slot capacity,
reroll buffer), and (c) retrying with corrective feedback when an
illegal pick comes back. There are no curated joker tier lists, no
pre-tagged item categories, and no scoring heuristics in the shop
loop — every shop / pack decision goes through the LLM.

Decisions stream out as schema-enforced JSON with a `scratchpad`
field that doubles as a chain-of-thought trace; an audit log
(`logs/shop_advisor.jsonl` by default) captures every decision with
its context for offline review.
