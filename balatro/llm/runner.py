"""Ollama HTTP client for the shop advisor.

Why a hand-rolled client and not the `ollama` Python package?
  - Zero extra deps in the project.
  - We need fine-grained control over timeout / format-schema / streaming.
  - Mirrors the pattern already used in llm_advisor.py for blackjack.

Usage:
    runner = OllamaRunner(model="qwen3:4b-instruct")
    raw = runner.chat(system="...", user="...", schema=SHOP_MAIN_SCHEMA)
    # raw is a dict matching the schema, or runner raises if all retries fail.
"""
from __future__ import annotations
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class CallStats:
    latency_s: float
    eval_count: int
    prompt_eval_count: int
    raw_response: str
    parsed: dict | None
    error: str | None


class OllamaRunner:
    def __init__(
        self,
        model: str = "qwen3:4b-instruct",
        url: str = "http://localhost:11434",
        timeout: float = 15.0,
        temperature: float = 0.0,
        num_predict: int = 1500,
        max_retries: int = 1,
        keep_alive: str = "30m",
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_predict = num_predict
        self.max_retries = max_retries
        self.keep_alive = keep_alive
        self.last_stats: CallStats | None = None

    def chat(
        self,
        system: str,
        user: str,
        schema: dict | None = None,
    ) -> dict:
        """Send a chat request, return parsed JSON dict.

        On parse/connection error, retry up to max_retries. Final failure
        raises RuntimeError — caller decides fallback (typically Skip).
        """
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_once(system, user, schema)
            except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.2)
                continue
        raise RuntimeError(f"Ollama call failed after {self.max_retries + 1} attempts: {last_err!r}")

    def _call_once(self, system: str, user: str, schema: dict | None) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                # 8K context — system prompt is ~1.8K tokens, user ~0.5K,
                # output up to 1.5K, leaving headroom for retries/menus.
                "num_ctx": 8192,
            },
        }
        if schema is not None:
            body["format"] = schema
        else:
            body["format"] = "json"

        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body_bytes = resp.read()
        latency = time.perf_counter() - t0

        envelope = json.loads(body_bytes)
        msg = envelope["message"]["content"]

        # Strip Qwen3 thinking blocks if /no_think was ignored.
        if "<think>" in msg and "</think>" in msg:
            tail = msg.split("</think>", 1)[1]
            msg = tail.strip() or msg

        parsed = json.loads(msg)

        self.last_stats = CallStats(
            latency_s=latency,
            eval_count=envelope.get("eval_count", 0),
            prompt_eval_count=envelope.get("prompt_eval_count", 0),
            raw_response=msg,
            parsed=parsed,
            error=None,
        )
        return parsed

    # ── diagnostics ─────────────────────────────────────────────────────

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=2.0) as r:
                return r.status == 200
        except Exception:
            return False
