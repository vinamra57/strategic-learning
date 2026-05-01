import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['C', 'D', 'H', 'S']
RANK_CHIPS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

HAND_BASE = {
    'high_card':      (5,   1),
    'pair':           (10,  2),
    'two_pair':       (20,  2),
    'three_of_kind':  (30,  3),
    'straight':       (30,  4),
    'flush':          (35,  4),
    'full_house':     (40,  4),
    'four_of_kind':   (60,  7),
    'straight_flush': (100, 8),
}

HAND_QUALITY = {
    'high_card':      0.1,
    'pair':           0.4,
    'two_pair':       0.6,
    'three_of_kind':  0.7,
    'straight':       0.9,
    'flush':          0.9,
    'full_house':     1.0,
    'four_of_kind':   1.0,
    'straight_flush': 1.0,
}


def make_deck():
    return [(r, s) for r in range(13) for s in range(4)]


def evaluate_hand(cards):
    """cards: list of (rank_idx, suit_idx). Returns (hand_name, score)."""
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    counts = sorted(rank_counts.values(), reverse=True)

    is_flush = len(cards) == 5 and len(set(suits)) == 1
    unique = sorted(set(ranks))
    is_straight = (
        len(cards) == 5
        and len(unique) == 5
        and (unique[-1] - unique[0] == 4 or unique == [0, 1, 2, 3, 12])
    )

    if is_straight and is_flush:
        name = 'straight_flush'
    elif counts[0] == 4:
        name = 'four_of_kind'
    elif counts[0] == 3 and len(counts) > 1 and counts[1] == 2:
        name = 'full_house'
    elif is_flush:
        name = 'flush'
    elif is_straight:
        name = 'straight'
    elif counts[0] == 3:
        name = 'three_of_kind'
    elif counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        name = 'two_pair'
    elif counts[0] == 2:
        name = 'pair'
    else:
        name = 'high_card'

    base_chips, mult = HAND_BASE[name]
    card_chips = sum(RANK_CHIPS[r] for r in ranks)
    return name, (base_chips + card_chips) * mult


class BalatroPLayEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(
        self,
        hand_size=8,
        max_hands=4,
        max_discards=4,
        target_score=800,
        render_mode="console",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.hand_size = hand_size
        self.max_hands = max_hands
        self.max_discards = max_discards
        self.target_score = target_score

        self._subsets = [
            list(combo)
            for size in range(1, 6)
            for combo in combinations(range(hand_size), size)
        ]
        self._n_subsets = len(self._subsets)
        self.action_space = spaces.Discrete(2 * self._n_subsets)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(hand_size * 2 + 3,),
            dtype=np.float32,
        )

    def _obs(self):
        ranks = np.array([c[0] / 12.0 for c in self.hand], dtype=np.float32)
        suits = np.array([c[1] / 3.0 for c in self.hand], dtype=np.float32)
        return np.concatenate([
            ranks, suits,
            [
                min(self.score / self.target_score, 1.0),
                self.hands_remaining / self.max_hands,
                self.discards_remaining / self.max_discards,
            ],
        ])

    def _draw(self, n):
        drawn = []
        for _ in range(n):
            if not self.deck:
                self.deck = make_deck()
                self.np_random.shuffle(self.deck)
            drawn.append(self.deck.pop())
        return drawn

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.deck = make_deck()
        self.np_random.shuffle(self.deck)
        self.score = 0
        self.hands_remaining = self.max_hands
        self.discards_remaining = self.max_discards
        self.hand = self._draw(self.hand_size)
        return self._obs(), {}

    def action_masks(self):
        masks = np.ones(2 * self._n_subsets, dtype=bool)
        if self.hands_remaining == 0:
            masks[: self._n_subsets] = False
        if self.discards_remaining == 0:
            masks[self._n_subsets :] = False
        return masks

    def step(self, action):
        is_discard = action >= self._n_subsets
        indices = self._subsets[action % self._n_subsets]
        selected = [self.hand[i] for i in indices]

        if is_discard:
            if self.discards_remaining <= 0:
                return self._obs(), -0.1, False, False, {"invalid": True}
            self.discards_remaining -= 1
            for i in sorted(indices, reverse=True):
                self.hand.pop(i)
            self.hand.extend(self._draw(len(indices)))
            return self._obs(), 0.0, False, False, {}

        # play
        if self.hands_remaining <= 0:
            return self._obs(), -0.1, False, False, {"invalid": True}
        self.hands_remaining -= 1
        hand_name, hand_score = evaluate_hand(selected)
        self.score += hand_score
        for i in sorted(indices, reverse=True):
            self.hand.pop(i)
        self.hand.extend(self._draw(len(indices)))

        if self.score >= self.target_score:
            return self._obs(), 1.0, True, False, {"hand": hand_name, "score": self.score, "won": True}
        if self.hands_remaining == 0:
            return self._obs(), -1.0, True, False, {"hand": hand_name, "score": self.score, "won": False}

        n_cards = len(indices)
        size_bonus = n_cards / 5.0
        quality = HAND_QUALITY.get(hand_name, 0.5)
        shaped_reward = (hand_score / self.target_score) * size_bonus * quality
        return self._obs(), shaped_reward, False, False, {"hand": hand_name}

    def render(self):
        if self.render_mode != "console":
            return
        hand_str = "  ".join(f"{RANKS[r]}{SUITS[s]}" for r, s in self.hand)
        print(f"[{hand_str}]")
        print(f"Score {self.score}/{self.target_score}  |  Hands {self.hands_remaining}  Discards {self.discards_remaining}")

    def close(self):
        pass


from stable_baselines3.common.callbacks import BaseCallback

class ScoreCallback(BaseCallback):
    def __init__(self, print_freq=2000):
        super().__init__()
        self.print_freq = print_freq
        self.episode_count = 0
        self.recent_scores = []
        self.logged_avgs = []
        self.wins = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            if "score" in info:
                self.recent_scores.append(info["score"])
                self.wins += int(info["won"])
                self.episode_count += 1
                if self.episode_count % self.print_freq == 0:
                    avg = sum(self.recent_scores[-self.print_freq:]) / self.print_freq
                    win_rate = self.wins / self.episode_count * 100
                    self.logged_avgs.append(round(avg, 1))
                    print(f"Episode {self.episode_count} | avg score {avg:.0f}/800 | win rate {win_rate:.1f}%")
                    json.dump(self.logged_avgs, open("scores.json", "w"))
        return True


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

    env = make_vec_env(BalatroPLayEnv, n_envs=4)
    model = PPO("MlpPolicy", env, verbose=0)

    checkpoint_cb = CheckpointCallback(
        save_freq=200_000,
        save_path="./checkpoints/",
        name_prefix="balatro_ppo",
    )
    score_cb = ScoreCallback(print_freq=2000)

    model.learn(
        total_timesteps=15_000_000,
        callback=CallbackList([score_cb, checkpoint_cb]),
    )
    model.save("balatro_ppo")

    avgs = score_cb.logged_avgs
    plt.plot(avgs)
    plt.xlabel(f"Checkpoint (every {score_cb.print_freq} episodes)")
    plt.ylabel("Avg score (650 target)")
    plt.title("Balatro PPO Training Curve")
    plt.savefig("training_curve.png")
    plt.show()