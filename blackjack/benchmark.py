"""Quick benchmark & convergence test for model.py."""
import time
from model import Config, train

# ── Speed benchmark ───────────────────────────────────────────────────────────
print("⏱  Speed benchmark: 500 shoes …")
cfg = Config(n_episodes=500, log_every=500, eval_every=10_000)
t0 = time.time()
train(cfg)
elapsed = time.time() - t0
print(f"   → {elapsed:.1f}s  ({elapsed/500*1000:.1f} ms/shoe)\n")

# ── Short convergence check ───────────────────────────────────────────────────
print("📈  Convergence check: 5 000 shoes …")
cfg2 = Config(n_episodes=5_000, log_every=500, eval_every=1_000, eval_episodes=100)
train(cfg2)
