import torch
import math
from model import Config, train
import sys
# just wrap huber loss to panic
import torch.nn.functional as F
orig = F.huber_loss
def safe_huber(*args, **kwargs):
    res = orig(*args, **kwargs)
    if torch.isnan(res):
        print("NaN in huber_loss!")
        sys.exit(1)
    return res
F.huber_loss = safe_huber
cfg = Config(n_episodes=50, log_every=10)
train(cfg)
