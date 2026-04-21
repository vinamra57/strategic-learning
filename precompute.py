"""
Offline LLM lookup-table builder.

Samples random shoe compositions, queries a local Ollama model for each
unique rounded state, and writes results to llm_table.json.

Usage:
    python3 precompute.py                        # 2000 samples, default model
    python3 precompute.py --samples 5000
    python3 precompute.py --samples 5000 --out llm_table.json

The resulting JSON maps cache-key strings to [true_count_norm, bet_confidence,
strategy_flag] float arrays — the same 3-dim feature vector LLMAdvisor uses.
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request

import numpy as np

from sim import Shoe, RANKS, HI_LO_COUNT
from llm_advisor import LLMAdvisor, _build_prompt


# ---------------------------------------------------------------------------
# Random shoe-state sampler
# ---------------------------------------------------------------------------

def _random_shoe_state(num_decks: int = 6) -> Shoe:
    """Return a Shoe at a random mid-shoe position (10%-90% dealt)."""
    shoe = Shoe(num_decks=num_decks, penetration=1.0)
    total = num_decks * 52
    target = random.randint(int(total * 0.10), int(total * 0.90))
    for _ in range(target):
        if shoe.cards:
            shoe.deal()
    return shoe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Precompute LLM lookup table")
    parser.add_argument("--samples",  type=int,   default=2000,
                        help="Number of random shoe states to sample (default: 2000)")
    parser.add_argument("--decks",    type=int,   default=6,
                        help="Number of decks per shoe (default: 6)")
    parser.add_argument("--model",    type=str,   default="qwen2.5:7b")
    parser.add_argument("--url",      type=str,   default="http://localhost:11434")
    parser.add_argument("--timeout",  type=int,   default=30)
    parser.add_argument("--out",      type=str,   default="llm_table.json")
    args = parser.parse_args()

    advisor = LLMAdvisor(model=args.model, url=args.url, timeout=args.timeout)

    # 1. Collect unique cache keys from random shoe states
    print(f"Sampling {args.samples} random shoe states …")
    key_to_shoe: dict[tuple, Shoe] = {}
    for _ in range(args.samples):
        shoe = _random_shoe_state(args.decks)
        key = advisor._cache_key(shoe)
        if key not in key_to_shoe:
            key_to_shoe[key] = shoe

    n_unique = len(key_to_shoe)
    print(f"  → {n_unique} unique cache keys (from {args.samples} samples)\n")

    # 2. Query LLM for each unique key
    table: dict[str, list[float]] = {}
    errors = 0
    t0 = time.time()

    for i, (key, shoe) in enumerate(key_to_shoe.items()):
        system, user = _build_prompt(shoe)
        try:
            raw = advisor._call(system, user)
            feats = advisor._parse(raw)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError) as e:
            print(f"  [warn] key {i}: {e!r} — skipping")
            errors += 1
            continue

        table[json.dumps(key)] = feats.tolist()

        # Progress every 50 entries
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_unique - i - 1) / rate
            print(f"  {i+1}/{n_unique}  ({rate:.1f} q/s  ~{remaining/60:.1f} min left)")

    elapsed = time.time() - t0
    print(f"\nDone. {len(table)} entries written, {errors} errors, {elapsed:.0f}s total.")

    # 3. Write JSON
    with open(args.out, "w") as f:
        json.dump(table, f)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
