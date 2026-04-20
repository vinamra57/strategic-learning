"""
Reinforcement learning agent for blackjack using Double DQN.

Two sub-agents:
  PlayAgent  — decides hit / stand / double during a hand
  MetaAgent  — decides bet size before each hand (including $0 to sit out)

Self-play: agents generate training data by playing episodes against
BlackjackEnv. An episode ends when the shoe is exhausted (truncated).
The objective is to maximise cumulative profit.

The agents are NOT given the card count. Instead they are given the remaining 
composition of the shoe (fractions of remaining 2s, 3s... etc). 
This allows the network to naturally deduce card counting via effect of removal.
"""

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sim import BlackjackEnv, Action
from llm_advisor import BaseAdvisor, NullAdvisor, LLMAdvisor

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Environment ──────────────────────────────────────────────────────────
    num_decks: int          = 6      # decks in the shoe (6 is standard)
    penetration: float      = 0.75   # fraction dealt before reshuffle
    starting_balance: float = 0.0    # chip stack at episode start
    min_bet: float          = 10.0   # table minimum
    max_bet: float          = 500.0  # table maximum
    n_bet_levels: int       = 5      # discrete bet sizes > 0 (sit-out is extra)
    allow_sit_out: bool     = True   # False = agent must bet every hand
    use_llm: bool           = False  # True = append LLM features to state (state_dim 13→16)
    llm_call_every: int     = 10     # call LLM every N hands, reuse features in between

    # ── Training ─────────────────────────────────────────────────────────────
    n_episodes: int         = 50_000 # shoes (episodes) to train on
    play_gamma: float       = 0.99   # PlayAgent discount factor
    meta_gamma: float       = 0.0    # MetaAgent discount factor (Contextual Bandit)
    lr: float               = 5e-4   # Adam learning rate (was initially 1e-3)
    batch_size: int         = 32
    replay_capacity: int    = 100_000
    target_update_freq: int = 2000   # online→target sync interval (steps)
    train_every: int        = 16     # Throttle .backward() frequency

    # ── Exploration ───────────────────────────────────────────────────────────
    eps_start: float        = 1.0
    eps_end: float          = 0.05
    eps_decay: int          = 10_000  # episodes (shoes) for epsilon to decay

    # ── Network ───────────────────────────────────────────────────────────────
    hidden_dim: int         = 128
    n_hidden_layers: int    = 2      # depth of each Q-net (excl. input/output)

    # ── Policy Gradient (PGPlayAgent) ─────────────────────────────────────────
    agent_type: str         = "dqn"  # "dqn" or "pg"
    pg_entropy_coef: float  = 0.05   # entropy regularisation for REINFORCE (was initially 0.01)
    pg_use_critic: bool     = True   # False = pure REINFORCE (no value baseline)

    # ── Logging / checkpointing ───────────────────────────────────────────────
    log_every: int          = 1_000  # episodes between training log lines
    eval_every: int         = 5_000  # episodes between evaluation runs
    eval_episodes: int      = 200    # greedy eval episodes
    save_path: str          = ""     # if set, save checkpoints here
    plot_path: str          = "training_curve.png"  # where to write the plot


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf: deque = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self._buf.append((s, a, r, ns, done))

    def sample(self, n: int):
        batch = random.sample(self._buf, n)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(a,            dtype=torch.long),
            torch.tensor(r,            dtype=torch.float32),
            torch.tensor(np.array(ns), dtype=torch.float32),
            torch.tensor(d,            dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────────────────────────────────────────

class QNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent (generic)
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """Double DQN with experience replay and a target network (for PlayAgent)."""

    def __init__(self, state_dim: int, n_actions: int, cfg: Config, gamma: float):
        self.n_actions = n_actions
        self.cfg = cfg
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = 0  # track episodes for epsilon decay

        self.q = QNet(state_dim, n_actions, cfg.hidden_dim,
                      cfg.n_hidden_layers).to(self.device)
        self.q_target = QNet(state_dim, n_actions, cfg.hidden_dim,
                             cfg.n_hidden_layers).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.replay_capacity)
        self.steps = 0
        self.losses: list[float] = []

    def epsilon(self) -> float:
        cfg = self.cfg
        return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(
            -self.episodes / cfg.eps_decay
        )

    def act(self,
            state: np.ndarray,
            legal_mask: list[int] | None = None,
            greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            pool = legal_mask if legal_mask is not None else list(range(self.n_actions))
            return random.choice(pool)

        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q(s).squeeze(0)

        if legal_mask is not None:
            mask = torch.full((self.n_actions,), float("-inf"), device=self.device)
            for a in legal_mask:
                mask[a] = 0.0
            q_vals = q_vals + mask

        return int(q_vals.argmax().item())

    def store(self, s, a, r, ns, done: bool):
        self.buf.push(s, a, r, ns, done)
        self.steps += 1

    def learn(self):
        if len(self.buf) < self.cfg.batch_size:
            return
        if self.steps % self.cfg.train_every != 0:
            return

        s, a, r, ns, d = [t.to(self.device) for t in
                          self.buf.sample(self.cfg.batch_size)]

        q_cur = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN
        with torch.no_grad():
            next_a = self.q(ns).argmax(1, keepdim=True)
            q_next = self.q_target(ns).gather(1, next_a).squeeze(1)

        target = r + self.gamma * q_next * (1.0 - d)

        loss = nn.functional.huber_loss(q_cur, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        self.losses.append(loss.item())

        if self.steps % self.cfg.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str | Path):
        torch.save({"q": self.q.state_dict(),
                    "q_target": self.q_target.state_dict(),
                    "opt": self.opt.state_dict(),
                    "steps": self.steps}, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.opt.load_state_dict(ckpt["opt"])
        self.steps = ckpt["steps"]


# ─────────────────────────────────────────────────────────────────────────────
# Bandit Agent (MetaAgent: no temporal credit, direct EV regression)
# ─────────────────────────────────────────────────────────────────────────────

class BanditAgent:
    """Contextual Bandit for bet sizing.
    
    Rather than Q-learning (which adds spurious multi-step credit), this agent
    learns a direct mapping: state -> expected_reward_per_unit_bet for each
    bet level (plus 0 for sit-out). It uses a simple MSE regression loss and no
    target network, converging much faster to the correct EV surface.
    """

    def __init__(self, state_dim: int, n_actions: int, cfg: Config):
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = 0  # for epsilon

        # Shallow net: state -> EV estimate per action
        self.net = QNet(state_dim, n_actions, cfg.hidden_dim,
                        cfg.n_hidden_layers).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.replay_capacity)
        self.steps = 0
        self.losses: list[float] = []

    def epsilon(self) -> float:
        cfg = self.cfg
        return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(
            -self.episodes / cfg.eps_decay
        )

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions)

        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ev = self.net(s).squeeze(0)
        return int(ev.argmax().item())

    def store(self, state, action: int, reward_per_unit: float):
        """Store (s, a, r/unit) — no next-state needed for bandit."""
        # Store as (s, a, r, s, False) reusing ReplayBuffer; ns unused.
        self.buf.push(state, action, reward_per_unit, state, False)
        self.steps += 1

    def learn(self):
        if len(self.buf) < self.cfg.batch_size:
            return
        if self.steps % self.cfg.train_every != 0:
            return

        s, a, r, _, _ = [t.to(self.device) for t in
                         self.buf.sample(self.cfg.batch_size)]

        # Predict EV for each action; update only the chosen action's head
        ev_pred = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.huber_loss(ev_pred, r)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.losses.append(loss.item())

    def save(self, path: str | Path):
        torch.save({"net": self.net.state_dict(),
                    "opt": self.opt.state_dict(),
                    "steps": self.steps}, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.steps = ckpt["steps"]


# ─────────────────────────────────────────────────────────────────────────────
# Policy Gradient Agent (REINFORCE + value baseline)
# ─────────────────────────────────────────────────────────────────────────────

class PGPlayAgent:
    """REINFORCE with a learned value-function baseline (actor-critic style).

    Unlike DQN the policy is stochastic: during training the agent samples
    from its softmax distribution, which provides built-in exploration.
    Entropy regularisation (cfg.pg_entropy_coef) prevents premature collapse.

    Update cadence: one gradient step per hand (trajectory), not per step.
    The per-step reward from the environment is already sparse (only the final
    step of a hand carries a non-zero reward), so accumulating the full
    hand trajectory before updating is both correct and efficient.

    API is intentionally compatible with DQNAgent so that run_episode /
    evaluate / train can use either agent without modification.
    """

    def __init__(self, state_dim: int, n_actions: int, cfg: Config, gamma: float):
        self.n_actions = n_actions
        self.cfg = cfg
        self.gamma = gamma
        self.use_critic = cfg.pg_use_critic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = 0   # for logging compatibility

        self.actor = QNet(state_dim, n_actions, cfg.hidden_dim, cfg.n_hidden_layers).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr)

        if self.use_critic:
            self.critic = QNet(state_dim, 1, cfg.hidden_dim, cfg.n_hidden_layers).to(self.device)
            self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # Hand-level trajectory buffer: list of (state, action, reward, done)
        self._traj: list[tuple] = []

        self.steps = 0
        self.losses: list[float] = []

    # ── epsilon stub (DQNAgent compat — PG uses stochastic policy instead) ──

    def epsilon(self) -> float:
        return 0.0

    # ── action selection ─────────────────────────────────────────────────────

    def act(self,
            state: np.ndarray,
            legal_mask: list[int] | None = None,
            greedy: bool = False) -> int:
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(s).squeeze(0)

        if legal_mask is not None:
            mask = torch.full((self.n_actions,), float("-inf"), device=self.device)
            for a in legal_mask:
                mask[a] = 0.0
            logits = logits + mask

        if greedy:
            return int(logits.argmax().item())

        probs = torch.softmax(logits, dim=-1)
        return int(torch.distributions.Categorical(probs).sample().item())

    # ── trajectory storage ───────────────────────────────────────────────────

    def store(self, s, a, r, _ns, done: bool):
        self._traj.append((np.array(s, dtype=np.float32), int(a), float(r), done))
        self.steps += 1

    # ── learning ─────────────────────────────────────────────────────────────

    def learn(self):
        """Run a gradient update if the most-recently stored step is terminal."""
        if not self._traj or not self._traj[-1][3]:
            return  # hand not yet finished

        # ── compute discounted returns G_t ───────────────────────────────────
        returns: list[float] = []
        G = 0.0
        for _, _, r, _ in reversed(self._traj):
            G = r + self.gamma * G
            returns.insert(0, G)

        states_np  = np.array([t[0] for t in self._traj])
        actions_np = np.array([t[1] for t in self._traj])

        states  = torch.tensor(states_np,  dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions_np, dtype=torch.long).to(self.device)
        returns_t = torch.tensor(returns,  dtype=torch.float32).to(self.device)

        if len(returns) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # ── critic (value baseline) update ────────────────────────────────────
        if self.use_critic:
            values = self.critic(states).squeeze(1)
            critic_loss = nn.functional.mse_loss(values, returns_t.detach())
            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_opt.step()

            with torch.no_grad():
                baseline = self.critic(states).squeeze(1)
            advantage = returns_t - baseline
        else:
            # Pure REINFORCE: normalized returns are already ~zero-mean, no baseline
            advantage = returns_t

        logits     = self.actor(states)
        log_probs  = torch.log_softmax(logits, dim=-1)
        sel_lp     = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy    = -(torch.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()

        actor_loss = -(sel_lp * advantage).mean() - self.cfg.pg_entropy_coef * entropy

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        self.losses.append(actor_loss.item())
        self._traj.clear()

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        d = {
            "actor":     self.actor.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "steps":     self.steps,
        }
        if self.use_critic:
            d["critic"]     = self.critic.state_dict()
            d["critic_opt"] = self.critic_opt.state_dict()
        torch.save(d, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.steps = ckpt["steps"]
        if self.use_critic and "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
            self.critic_opt.load_state_dict(ckpt["critic_opt"])


def get_remaining_ratios(shoe) -> np.ndarray:
    """Returns a 10D vector of remaining shoe composition (raw counts divided by starting counts).
    This allows the NN to assign weights directly to specific cards (e.g. 10s and As = valuable)."""
    counts = {str(i): 0 for i in range(2, 11)}
    counts['A'] = 0
    for c in shoe.cards: 
        if c in ['J', 'Q', 'K']:
            counts['10'] += 1
        else:
            counts[c] += 1
            
    num_decks = shoe.num_decks
    ratios = [
        counts['2'] / (4 * num_decks),
        counts['3'] / (4 * num_decks),
        counts['4'] / (4 * num_decks),
        counts['5'] / (4 * num_decks),
        counts['6'] / (4 * num_decks),
        counts['7'] / (4 * num_decks),
        counts['8'] / (4 * num_decks),
        counts['9'] / (4 * num_decks),
        counts['10'] / (16 * num_decks),
        counts['A'] / (4 * num_decks),
    ]
    return np.array(ratios, dtype=np.float32)

def encode_play(obs: tuple, shoe) -> np.ndarray:
    """13-dim state for PlayAgent."""
    player_sum, dealer_up, usable_ace = obs
    ratios = get_remaining_ratios(shoe)
    base = np.array([
        player_sum / 21.0,
        dealer_up / 11.0,
        float(usable_ace),
    ], dtype=np.float32)
    return np.concatenate([ratios, base])


def encode_meta(shoe, balance: float, cfg: Config) -> np.ndarray:
    """13-dim state for MetaAgent."""
    ratios = get_remaining_ratios(shoe)
    base = np.array([
        shoe.decks_remaining / cfg.num_decks,
        shoe.cards_dealt / (shoe.num_decks * 52),
        balance / (cfg.min_bet * 100) if cfg.min_bet > 0 else 0.0,
    ], dtype=np.float32)
    return np.concatenate([ratios, base])


# ─────────────────────────────────────────────────────────────────────────────
# Bet level helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_bet_levels(min_bet: float, max_bet: float, n: int) -> list[float]:
    """n log-spaced bet amounts from min_bet to max_bet."""
    if n == 1:
        return [min_bet]
    return [
        round(min_bet * ((max_bet / min_bet) ** (i / (n - 1))), 2)
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env: BlackjackEnv,
    play_agent: DQNAgent,
    meta_agent: BanditAgent,
    bet_levels: list[float],
    cfg: Config,
    greedy: bool = False,
    learn: bool = True,
    advisor: "BaseAdvisor | None" = None,
) -> float:
    """Play one shoe. Returns total_profit.

    MetaAgent action space : 
        0             → Sit out (Bet $0)
        1 .. n_levels → Place bet `bet_levels[meta_action - 1]`
    
    PlayAgent action space  : 0=stand, 1=hit, 2=double
    """
    balance = cfg.starting_balance
    shoe = env.shoe
    shoe._build()
    _advisor = advisor or NullAdvisor()
    _llm_feats: np.ndarray | None = None
    _hand_count = 0

    while not shoe.needs_reshuffle():

        # ── LLM features (called every llm_call_every hands) ─────────────────
        if cfg.use_llm:
            if _hand_count % cfg.llm_call_every == 0:
                _llm_feats = _advisor.features(shoe)
            llm_feats = _llm_feats
        else:
            llm_feats = None
        _hand_count += 1

        # ── MetaAgent: bet sizing ────────────────────────────────────────────
        ms_base = encode_meta(shoe, balance, cfg)
        ms = np.concatenate([ms_base, llm_feats]) if llm_feats is not None else ms_base
        meta_action = meta_agent.act(ms, greedy=greedy)

        if cfg.allow_sit_out and meta_action == 0:
            bet = 0.0
        elif cfg.allow_sit_out:
            bet = bet_levels[meta_action - 1]
        else:
            bet = bet_levels[meta_action]

        # ── Deal a hand ───────────────────────────────────────────────────────
        is_sitting_out = cfg.allow_sit_out and (meta_action == 0)
        virtual_bet = cfg.min_bet if is_sitting_out else bet
        
        obs, info = env.reset(bet=virtual_bet)

        # Dealer blackjack — settled before player acts
        if info["dealer_blackjack"]:
            raw_reward = env._dealer_play()
            real_reward = 0.0 if is_sitting_out else raw_reward
            balance += real_reward
            if learn:
                _store_meta(meta_agent, ms, meta_action, real_reward, bet, cfg)
            continue

        # Player natural blackjack (dealer confirmed no BJ)
        if info["player_hand"].is_blackjack:
            raw_reward = env._dealer_play()
            real_reward = 0.0 if is_sitting_out else raw_reward
            balance += real_reward
            if learn:
                _store_meta(meta_agent, ms, meta_action, real_reward, bet, cfg)
            continue

        # ── PlayAgent: in-hand decisions ──────────────────────────────────────
        ps_base = encode_play(obs, shoe)
        ps = np.concatenate([ps_base, llm_feats]) if llm_feats is not None else ps_base
        terminated = False
        hand_reward = 0.0

        while not terminated:
            legal = [int(a) for a in env.legal_actions()]
            play_action = play_agent.act(ps, legal_mask=legal, greedy=greedy)

            action_enum = Action(play_action)
            obs, raw_r, terminated, _, info = env.step(action_enum)

            play_reward = raw_r / virtual_bet
            next_ps_base = encode_play(obs, shoe)
            next_ps = np.concatenate([next_ps_base, llm_feats]) if llm_feats is not None else next_ps_base

            if learn:
                play_agent.store(ps, play_action, play_reward, next_ps, terminated)
                play_agent.learn()

            ps = next_ps
            hand_reward = raw_r

        real_hand_reward = 0.0 if is_sitting_out else hand_reward
        balance += real_hand_reward
        if learn:
            _store_meta(meta_agent, ms, meta_action, real_hand_reward, bet, cfg)

    return balance - cfg.starting_balance


def _store_meta(meta_agent: "BanditAgent", ms, action, hand_profit, bet, cfg):
    """Store a BanditAgent transition: reward is profit divided by bet (EV/unit)."""
    r = (hand_profit / bet) if bet > 0 else 0.0
    meta_agent.store(ms, action, r)
    meta_agent.learn()

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    play_agent: DQNAgent,
    meta_agent: BanditAgent,
    bet_levels: list[float],
    cfg: Config,
    advisor: "BaseAdvisor | None" = None,
) -> dict:
    env = BlackjackEnv(cfg.num_decks, cfg.penetration)
    _advisor = advisor or NullAdvisor()
    profits = []
    
    # We will also calculate sit_out rate: proportion of hands where bet=0.
    # To do this cleanly, we just log meta agent actions.
    sit_outs = 0
    hands = 0
    hands_won = 0
    
    for _ in range(cfg.eval_episodes):
        shoe = env.shoe
        shoe._build()
        balance = 0.0
        
        _llm_feats_eval: np.ndarray | None = None
        _hand_count_eval = 0
        while not shoe.needs_reshuffle():
            hands += 1
            if cfg.use_llm:
                if _hand_count_eval % cfg.llm_call_every == 0:
                    _llm_feats_eval = _advisor.features(shoe)
                llm_feats = _llm_feats_eval
            else:
                llm_feats = None
            _hand_count_eval += 1
            ms_base = encode_meta(shoe, balance, cfg)
            ms = np.concatenate([ms_base, llm_feats]) if llm_feats is not None else ms_base
            meta_action = meta_agent.act(ms, greedy=True)
            if cfg.allow_sit_out and meta_action == 0:
                sit_outs += 1
                bet = 0.0
            elif cfg.allow_sit_out:
                bet = bet_levels[meta_action - 1]
            else:
                bet = bet_levels[meta_action]

            is_sitting_out = cfg.allow_sit_out and (meta_action == 0)
            virtual_bet = cfg.min_bet if is_sitting_out else bet
            
            obs, info = env.reset(bet=virtual_bet)
                
            if info["dealer_blackjack"]:
                real_reward = 0.0 if is_sitting_out else env._dealer_play()
                if real_reward > 0:
                    hands_won += 1
                balance += real_reward
                continue
            if info["player_hand"].is_blackjack:
                real_reward = 0.0 if is_sitting_out else env._dealer_play()
                if real_reward > 0:
                    hands_won += 1
                balance += real_reward
                continue
                
            ps_base = encode_play(obs, shoe)
            ps = np.concatenate([ps_base, llm_feats]) if llm_feats is not None else ps_base
            terminated = False
            while not terminated:
                legal = [int(a) for a in env.legal_actions()]
                play_action = play_agent.act(ps, legal_mask=legal, greedy=True)
                obs, raw_r, terminated, _, info = env.step(Action(play_action))
                ps_base = encode_play(obs, shoe)
                ps = np.concatenate([ps_base, llm_feats]) if llm_feats is not None else ps_base
            real_reward = 0.0 if is_sitting_out else raw_r
            if real_reward > 0:
                hands_won += 1
            balance += real_reward
            
        profits.append(balance)

    arr = np.array(profits)
    return {
        "mean":          float(arr.mean()),
        "std":           float(arr.std()),
        "median":        float(np.median(arr)),
        "win_rate":      float((arr > 0).mean()),
        "min":           float(arr.min()),
        "max":           float(arr.max()),
        "sit_out_rate":  float(sit_outs / hands) if hands > 0 else 0.0,
        "hand_win_rate": float(hands_won / (hands - sit_outs)) if (hands - sit_outs) > 0 else 0.0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg: Config,
    play_agent: "DQNAgent | PGPlayAgent | None" = None,
    meta_agent: "BanditAgent | None" = None,
    advisor: "BaseAdvisor | None" = None,
) -> tuple["DQNAgent | PGPlayAgent", "BanditAgent"]:
    bet_levels     = build_bet_levels(cfg.min_bet, cfg.max_bet, cfg.n_bet_levels)
    n_meta_actions = len(bet_levels) + (1 if cfg.allow_sit_out else 0)

    pg_mode = f"{cfg.agent_type.upper()}" + (
        " + critic" if cfg.agent_type == "pg" and cfg.pg_use_critic else
        " (no critic)" if cfg.agent_type == "pg" else ""
    )
    print(f"Play agent          : {pg_mode}")
    sit_label = " (plus $0.0 for Sit Out)" if cfg.allow_sit_out else " (sit-out disabled)"
    print(f"Bet levels          : {bet_levels}{sit_label}")
    print(f"Meta actions        : {n_meta_actions}")
    print(f"Device              : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"LLM advisor         : {'Qwen (state_dim=16)' if cfg.use_llm else 'disabled (state_dim=13)'}")
    print()

    _advisor = advisor or (LLMAdvisor() if cfg.use_llm else NullAdvisor())
    state_dim = 13 + (_advisor.FEATURE_DIM if cfg.use_llm else 0)

    if play_agent is None:
        if cfg.agent_type == "pg":
            play_agent = PGPlayAgent(state_dim=state_dim, n_actions=3, cfg=cfg, gamma=cfg.play_gamma)
        else:
            play_agent = DQNAgent(state_dim=state_dim, n_actions=3, cfg=cfg, gamma=cfg.play_gamma)
    if meta_agent is None:
        meta_agent = BanditAgent(state_dim=state_dim, n_actions=n_meta_actions, cfg=cfg)

    env = BlackjackEnv(cfg.num_decks, cfg.penetration)

    profit_window:     deque[float] = deque(maxlen=cfg.log_every)

    train_log: list[dict] = []
    eval_log:  list[dict] = []

    for ep in range(1, cfg.n_episodes + 1):
        profit = run_episode(env, play_agent, meta_agent, bet_levels, cfg, advisor=_advisor)
        profit_window.append(profit)

        # Tick episode counters for epsilon decay
        play_agent.episodes += 1
        meta_agent.episodes += 1

        if ep % cfg.log_every == 0:
            avg    = float(np.mean(profit_window))
            std    = float(np.std(profit_window))
            eps    = play_agent.epsilon()
            p_loss = float(np.mean(play_agent.losses[-200:])) if play_agent.losses else float("nan")
            m_loss = float(np.mean(meta_agent.losses[-200:])) if meta_agent.losses else float("nan")

            train_log.append({
                "episode":       ep,
                "mean_profit":   avg,
                "std_profit":    std,
            })

            print(
                f"ep {ep:>8,}  avg_profit={avg:>+8.2f}  std={std:>7.2f}"
                f"  ε={eps:.3f}"
                f"  play_loss={p_loss:.4f}  meta_loss={m_loss:.4f}"
            )

        if ep % cfg.eval_every == 0:
            stats = evaluate(play_agent, meta_agent, bet_levels, cfg, advisor=_advisor)
            stats["episode"] = ep
            eval_log.append(stats)
            print(
                f"  ┌─ EVAL ({cfg.eval_episodes} shoes, greedy) ─"
                f"\n  │  mean={stats['mean']:>+8.2f}  std={stats['std']:.2f}"
                f"\n  │  median={stats['median']:>+7.2f}  win_rate={stats['win_rate']:.1%}"
                f"\n  │  sit_out_rate={stats['sit_out_rate']:.1%}  hand_win_rate={stats['hand_win_rate']:.1%}"
                f"\n  └─ range [{stats['min']:+.2f}, {stats['max']:+.2f}]"
            )

    if cfg.save_path:
        p = Path(cfg.save_path)
        p.mkdir(parents=True, exist_ok=True)
        play_agent.save(p / "play_agent.pt")
        meta_agent.save(p / "meta_agent.pt")
        print(f"\nCheckpoints saved → {p}/")

    return play_agent, meta_agent


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Double-DQN blackjack agent that learns to count implicitly (PlayAgent + MetaAgent)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = Config()

    env = p.add_argument_group("environment")
    env.add_argument("--num-decks",        type=int,   default=d.num_decks)
    env.add_argument("--penetration",      type=float, default=d.penetration)
    env.add_argument("--starting-balance", type=float, default=d.starting_balance)
    env.add_argument("--min-bet",          type=float, default=d.min_bet)
    env.add_argument("--max-bet",          type=float, default=d.max_bet)
    env.add_argument("--n-bet-levels",     type=int,   default=d.n_bet_levels)
    env.add_argument("--no-sit-out",       dest="allow_sit_out", action="store_false",
                     help="Force agent to bet every hand (no sit-out action)")
    env.add_argument("--use-llm",          dest="use_llm", action="store_true",
                     help="Append Qwen LLM card-count features to state (requires ollama serve)")
    env.add_argument("--llm-call-every",   type=int, default=Config().llm_call_every,
                     help="Call LLM every N hands, reuse features in between")

    tr = p.add_argument_group("training")
    tr.add_argument("--n-episodes",         type=int,   default=d.n_episodes)
    tr.add_argument("--play-gamma",         type=float, default=d.play_gamma)
    tr.add_argument("--meta-gamma",         type=float, default=d.meta_gamma)
    tr.add_argument("--lr",                 type=float, default=d.lr)
    tr.add_argument("--batch-size",         type=int,   default=d.batch_size)
    tr.add_argument("--replay-capacity",    type=int,   default=d.replay_capacity)
    tr.add_argument("--target-update-freq", type=int,   default=d.target_update_freq)
    tr.add_argument("--train-every",        type=int,   default=d.train_every)

    ex = p.add_argument_group("exploration")
    ex.add_argument("--eps-start",          type=float, default=d.eps_start)
    ex.add_argument("--eps-end",            type=float, default=d.eps_end)
    ex.add_argument("--eps-decay",          type=int,   default=d.eps_decay)

    nn_ = p.add_argument_group("network")
    nn_.add_argument("--hidden-dim",        type=int,   default=d.hidden_dim)
    nn_.add_argument("--n-hidden-layers",   type=int,   default=d.n_hidden_layers)

    pg = p.add_argument_group("policy gradient")
    pg.add_argument("--agent-type",         type=str,   default=d.agent_type,
                    choices=["dqn", "pg"],
                    help="Play agent algorithm: dqn (Double DQN) or pg (REINFORCE+baseline)")
    pg.add_argument("--pg-entropy-coef",    type=float, default=d.pg_entropy_coef)
    pg.add_argument("--no-critic",          dest="pg_use_critic", action="store_false",
                    help="Pure REINFORCE: disable value-function baseline")

    lg = p.add_argument_group("logging")
    lg.add_argument("--log-every",          type=int,   default=d.log_every)
    lg.add_argument("--eval-every",         type=int,   default=d.eval_every)
    lg.add_argument("--eval-episodes",      type=int,   default=d.eval_episodes)
    lg.add_argument("--save-path",          type=str,   default=d.save_path)
    lg.add_argument("--plot-path",          type=str,   default=d.plot_path)

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg = Config(
        num_decks               = args.num_decks,
        penetration             = args.penetration,
        starting_balance        = args.starting_balance,
        min_bet                 = args.min_bet,
        max_bet                 = args.max_bet,
        n_bet_levels            = args.n_bet_levels,
        allow_sit_out           = args.allow_sit_out,
        use_llm                 = args.use_llm,
        llm_call_every          = args.llm_call_every,
        n_episodes              = args.n_episodes,
        play_gamma              = args.play_gamma,
        meta_gamma              = args.meta_gamma,
        lr                      = args.lr,
        batch_size              = args.batch_size,
        replay_capacity         = args.replay_capacity,
        target_update_freq      = args.target_update_freq,
        train_every             = args.train_every,
        eps_start               = args.eps_start,
        eps_end                 = args.eps_end,
        eps_decay               = args.eps_decay,
        hidden_dim              = args.hidden_dim,
        n_hidden_layers         = args.n_hidden_layers,
        agent_type              = args.agent_type,
        pg_entropy_coef         = args.pg_entropy_coef,
        pg_use_critic           = args.pg_use_critic,
        log_every               = args.log_every,
        eval_every              = args.eval_every,
        eval_episodes           = args.eval_episodes,
        save_path               = args.save_path,
        plot_path               = args.plot_path,
    )
    
    train(cfg)
