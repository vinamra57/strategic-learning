from model import Config, train
import numpy as np

cfg = Config(n_episodes=5, log_every=1, eval_every=100)
play, meta = train(cfg)
print(f"Meta buf len: {len(meta.buf)}")
print(f"Play buf len: {len(play.buf)}")
print(f"Meta losses: {len(meta.losses)}")
print(f"Play losses: {len(play.losses)}")
print(f"EPS: {play.epsilon()}")
