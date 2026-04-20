"""
LLM card-counting advisor.

Wraps a local Qwen model (via Ollama REST API) to analyse shoe composition
and return a structured signal that gets appended to RL state vectors.

Feature vector  (LLMAdvisor.FEATURE_DIM = 3):
    [0]  estimated true count, clipped and normalised to [-1, 1]
    [1]  bet confidence  0 = sit-out / minimum,  1 = table max
    [2]  strategy flag   0 = conservative (stand-heavy), 1 = aggressive

During RL training the LLM is too slow to call every hand (~1-5 s/call).
Use NullAdvisor as a drop-in replacement that returns zeros.  Swap in
LLMAdvisor for evaluation or live play.

Quick start:
    advisor = LLMAdvisor()          # needs `ollama serve` running
    feats   = advisor.features(shoe)
    state   = np.concatenate([encode_play(obs, shoe), feats])
    # bump agent state_dim: 13 → 13 + LLMAdvisor.FEATURE_DIM
"""

import json
import urllib.request
import urllib.error
import numpy as np

from sim import Shoe


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a blackjack card-counting expert. You will be given the remaining
composition of a shoe as fractions of each rank still undealt.
Respond ONLY with a JSON object — no explanation, no markdown.
"""

_USER_TMPL = """\
Shoe composition (fraction of each rank remaining out of a full shoe):
  2s: {r2:.3f}  3s: {r3:.3f}  4s: {r4:.3f}  5s: {r5:.3f}  6s: {r6:.3f}
  7s: {r7:.3f}  8s: {r8:.3f}  9s: {r9:.3f}  10s/face: {r10:.3f}  Aces: {rA:.3f}

Decks remaining: {decks:.2f}
Cards dealt so far: {dealt} of {total}

Using Hi-Lo counting (2-6 = +1, 7-9 = 0, 10/face/A = -1):
1. Estimate the running count from the composition above.
2. Compute the true count (running count / decks remaining).
3. Recommend a bet confidence from 0.0 (sit out) to 1.0 (max bet).
4. Recommend a strategy: "conservative" or "aggressive".

Respond with exactly this JSON and nothing else:
{{
  "true_count": <float>,
  "bet_confidence": <float 0.0-1.0>,
  "strategy": "conservative" | "aggressive",
  "reasoning": "<one short sentence>"
}}
"""


def _build_prompt(shoe: Shoe) -> tuple[str, str]:
    nd = shoe.num_decks
    counts: dict[str, int] = {str(i): 0 for i in range(2, 11)}
    counts["A"] = 0
    for c in shoe.cards:
        key = "10" if c in ("J", "Q", "K") else c
        counts[key] += 1

    user = _USER_TMPL.format(
        r2=counts["2"]  / (4 * nd),
        r3=counts["3"]  / (4 * nd),
        r4=counts["4"]  / (4 * nd),
        r5=counts["5"]  / (4 * nd),
        r6=counts["6"]  / (4 * nd),
        r7=counts["7"]  / (4 * nd),
        r8=counts["8"]  / (4 * nd),
        r9=counts["9"]  / (4 * nd),
        r10=counts["10"] / (16 * nd),
        rA=counts["A"]  / (4 * nd),
        decks=shoe.decks_remaining,
        dealt=shoe.cards_dealt,
        total=nd * 52,
    )
    return _SYSTEM, user


# ─────────────────────────────────────────────────────────────────────────────
# Advisor base
# ─────────────────────────────────────────────────────────────────────────────

class BaseAdvisor:
    FEATURE_DIM = 3

    def features(self, shoe: Shoe) -> np.ndarray:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Null advisor  (training-time drop-in, zero overhead)
# ─────────────────────────────────────────────────────────────────────────────

class NullAdvisor(BaseAdvisor):
    """Returns a zero feature vector. Use during RL training."""

    def features(self, shoe: Shoe) -> np.ndarray:
        return np.zeros(self.FEATURE_DIM, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LLM advisor  (evaluation / live play)
# ─────────────────────────────────────────────────────────────────────────────

class LLMAdvisor(BaseAdvisor):
    """
    Calls a local Ollama model and parses its JSON response into a
    3-dim feature vector.

    Args:
        model:    Ollama model tag, e.g. "qwen2.5:7b"
        url:      Ollama API base URL
        timeout:  HTTP timeout in seconds
        fallback: feature vector to return on parse / connection failure
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        url: str = "http://localhost:11434",
        timeout: int = 30,
        fallback: np.ndarray | None = None,
    ):
        self.model   = model
        self.url     = url.rstrip("/")
        self.timeout = timeout
        self.fallback = fallback if fallback is not None else np.zeros(
            self.FEATURE_DIM, dtype=np.float32
        )
        self._cache: dict[tuple, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    # ── internal ─────────────────────────────────────────────────────────────

    def _call(self, system: str, user: str) -> dict:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "format": "json",
        }).encode()

        req = urllib.request.Request(
            f"{self.url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())

        return json.loads(body["message"]["content"])

    def _parse(self, raw: dict) -> np.ndarray:
        true_count     = float(raw.get("true_count", 0.0))
        bet_confidence = float(raw.get("bet_confidence", 0.5))
        strategy       = raw.get("strategy", "conservative")

        norm_count    = float(np.clip(true_count / 6.0, -1.0, 1.0))
        bet_conf      = float(np.clip(bet_confidence, 0.0, 1.0))
        strategy_flag = 1.0 if strategy == "aggressive" else 0.0

        return np.array([norm_count, bet_conf, strategy_flag], dtype=np.float32)

    # ── public ────────────────────────────────────────────────────────────────

    def _cache_key(self, shoe: Shoe) -> tuple:
        """Round shoe ratios to 1dp + decks_remaining to 0.5 for cache bucketing."""
        counts: dict[str, int] = {str(i): 0 for i in range(2, 11)}
        counts["A"] = 0
        for c in shoe.cards:
            key = "10" if c in ("J", "Q", "K") else c
            counts[key] += 1
        nd = shoe.num_decks
        ratios = (
            round(counts["2"]  / (4  * nd), 1),
            round(counts["3"]  / (4  * nd), 1),
            round(counts["4"]  / (4  * nd), 1),
            round(counts["5"]  / (4  * nd), 1),
            round(counts["6"]  / (4  * nd), 1),
            round(counts["7"]  / (4  * nd), 1),
            round(counts["8"]  / (4  * nd), 1),
            round(counts["9"]  / (4  * nd), 1),
            round(counts["10"] / (16 * nd), 1),
            round(counts["A"]  / (4  * nd), 1),
            round(shoe.decks_remaining * 2) / 2,  # nearest 0.5
        )
        return ratios

    def features(self, shoe: Shoe) -> np.ndarray:
        """
        Query the LLM about the current shoe state.
        Results are cached by rounded shoe composition to avoid redundant calls.
        Returns a 3-dim float32 array. Falls back to self.fallback on error.
        """
        key = self._cache_key(shoe)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key].copy()

        self.cache_misses += 1
        system, user = _build_prompt(shoe)
        try:
            raw = self._call(system, user)
            result = self._parse(raw)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError) as e:
            print(f"[LLMAdvisor] warning: {e!r} — using fallback features")
            result = self.fallback.copy()

        self._cache[key] = result
        return result.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Augmented state encoders
# ─────────────────────────────────────────────────────────────────────────────

def encode_play_llm(obs: tuple, shoe: Shoe, advisor: BaseAdvisor) -> np.ndarray:
    """13-dim base state + 3-dim LLM features = 16-dim."""
    from model import encode_play
    return np.concatenate([encode_play(obs, shoe), advisor.features(shoe)])


def encode_meta_llm(shoe: Shoe, balance: float, cfg, advisor: BaseAdvisor) -> np.ndarray:
    """13-dim base state + 3-dim LLM features = 16-dim."""
    from model import encode_meta
    return np.concatenate([encode_meta(shoe, balance, cfg), advisor.features(shoe)])
