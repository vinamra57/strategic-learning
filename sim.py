"""
Blackjack simulation with n-deck shoe.

Architecture:
  BlackjackEnv  — pure game logic, gym-style interface for RL
  InteractiveGame — thin CLI wrapper around BlackjackEnv

RL interface (BlackjackEnv):
  reset()  -> obs, info
  step(action) -> obs, reward, terminated, truncated, info

  Actions: 0=stand, 1=hit, 2=double, 3=split (split not yet implemented)
  Obs: (player_sum, dealer_upcard, usable_ace)
"""

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Action(IntEnum):
    STAND  = 0
    HIT    = 1
    DOUBLE = 2
    # SPLIT = 3  # placeholder for future implementation


CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11,
}

HI_LO_COUNT = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1,
}

RANKS = list(CARD_VALUES.keys())
SUITS = ['♠', '♥', '♦', '♣']

# Dealer peeks for blackjack when showing any of these
PEEK_RANKS = frozenset({'10', 'J', 'Q', 'K', 'A'})


# ---------------------------------------------------------------------------
# Shoe
# ---------------------------------------------------------------------------

class Shoe:
    """n-deck shoe with hi-lo running count tracking and penetration cut."""

    def __init__(self, num_decks: int = 6, penetration: float = 0.75):
        """
        Args:
            num_decks: number of decks in the shoe
            penetration: fraction of shoe dealt before reshuffling (0-1)
        """
        self.num_decks = num_decks
        self.penetration = penetration
        self._build()

    def _build(self):
        self.cards = [rank for rank in RANKS for _ in SUITS for _ in range(self.num_decks)]
        random.shuffle(self.cards)
        self.running_count = 0
        self.cards_dealt = 0
        self._cut = int(len(self.cards) * self.penetration)

    def deal(self) -> str:
        card = self.cards.pop()
        self.running_count += HI_LO_COUNT[card]
        self.cards_dealt += 1
        return card

    def needs_reshuffle(self) -> bool:
        return len(self.cards) == 0 or self.cards_dealt >= self._cut

    @property
    def decks_remaining(self) -> float:
        return max(len(self.cards) / 52, 0.5)

    @property
    def true_count(self) -> float:
        """Hi-lo true count (running count / decks remaining)."""
        return self.running_count / self.decks_remaining


# ---------------------------------------------------------------------------
# Hand
# ---------------------------------------------------------------------------

@dataclass
class Hand:
    cards: list[str] = field(default_factory=list)
    bet: float = 1.0
    doubled: bool = False

    def add(self, card: str):
        self.cards.append(card)

    @property
    def value(self) -> int:
        total = sum(CARD_VALUES[c] for c in self.cards)
        aces = self.cards.count('A')
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    @property
    def is_bust(self) -> bool:
        return self.value > 21

    @property
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.value == 21

    @property
    def usable_ace(self) -> bool:
        """True if hand contains an ace counted as 11."""
        total = sum(CARD_VALUES[c] for c in self.cards)
        aces = self.cards.count('A')
        # An ace is 'usable' (soft) if treating one as 11 doesn't bust
        return aces > 0 and total <= 21 and (total - 10) != self.value

    def __str__(self) -> str:
        return ' '.join(self.cards) + f'  [{self.value}]'


# ---------------------------------------------------------------------------
# BlackjackEnv  (RL-ready game logic)
# ---------------------------------------------------------------------------

class BlackjackEnv:
    """
    Gym-style Blackjack environment.

    Observation space (tuple):
        player_sum   : int  [4, 21]
        dealer_upcard: int  [2, 11]
        usable_ace   : bool

    Reward:
        +1   win, +1.5 blackjack (net of bet), 0 push, -1 loss
        If doubled, rewards are scaled by 2.
    """

    def __init__(self, num_decks: int = 6, penetration: float = 0.75):
        self.shoe = Shoe(num_decks, penetration)
        self.player_hand: Optional[Hand] = None
        self.dealer_hand: Optional[Hand] = None
        self._done = True
        self._dealer_blackjack = False

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, bet: float = 1.0) -> tuple[tuple, dict]:
        """Deal a new round. Returns (obs, info)."""
        self._done = False
        self._dealer_blackjack = False
        self.player_hand = Hand(bet=bet)
        self.dealer_hand = Hand()

        # Deal: player, dealer, player, dealer (standard order)
        self.player_hand.add(self.shoe.deal())
        self.dealer_hand.add(self.shoe.deal())
        self.player_hand.add(self.shoe.deal())
        self.dealer_hand.add(self.shoe.deal())

        upcard = self.dealer_hand.cards[0]

        # Dealer peeks when showing a 10-value card or Ace
        if upcard in PEEK_RANKS and self.dealer_hand.is_blackjack:
            self._dealer_blackjack = True
            self._done = True
        elif self.player_hand.is_blackjack:
            # Player BJ only matters if dealer doesn't also have BJ
            self._done = True

        return self._obs(), self._info()

    def step(self, action: int) -> tuple[tuple, float, bool, bool, dict]:
        """
        Apply action. Returns (obs, reward, terminated, truncated, info).
        truncated is always False (no time limit within a hand).
        """
        if self._done:
            raise RuntimeError("Call reset() before step().")

        action = Action(action)

        if action == Action.HIT:
            self.player_hand.add(self.shoe.deal())
            if self.player_hand.is_bust:
                self._done = True
                return self._obs(), -self.player_hand.bet, True, False, self._info()

        elif action == Action.DOUBLE:
            self.player_hand.bet *= 2
            self.player_hand.doubled = True
            self.player_hand.add(self.shoe.deal())
            # After double, player must stand
            if self.player_hand.is_bust:
                self._done = True
                return self._obs(), -self.player_hand.bet, True, False, self._info()
            # Fall through to dealer play
            action = Action.STAND

        if action == Action.STAND:
            reward = self._dealer_play()
            self._done = True
            return self._obs(), reward, True, False, self._info()

        # HIT and not bust — game continues
        return self._obs(), 0.0, False, False, self._info()

    def legal_actions(self) -> list[Action]:
        actions = [Action.STAND, Action.HIT]
        if len(self.player_hand.cards) == 2 and not self.player_hand.doubled:
            actions.append(Action.DOUBLE)
        return actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dealer_play(self) -> float:
        """Run dealer logic and return reward."""
        # Dealer hits on soft 17
        while self.dealer_hand.value < 17 or (
            self.dealer_hand.value == 17 and self.dealer_hand.usable_ace
        ):
            self.dealer_hand.add(self.shoe.deal())

        p = self.player_hand.value
        d = self.dealer_hand.value
        bet = self.player_hand.bet

        if self.player_hand.is_blackjack and not self.dealer_hand.is_blackjack:
            return 1.5 * bet
        if self.player_hand.is_blackjack and self.dealer_hand.is_blackjack:
            return 0.0  # push
        if self.dealer_hand.is_bust or p > d:
            return bet
        if p == d:
            return 0.0  # push
        return -bet

    def _obs(self) -> tuple:
        return (
            self.player_hand.value,
            CARD_VALUES[self.dealer_hand.cards[0]],
            self.player_hand.usable_ace,
        )

    def _info(self) -> dict:
        return {
            'player_hand': self.player_hand,
            'dealer_hand': self.dealer_hand,
            'running_count': self.shoe.running_count,
            'true_count': self.shoe.true_count,
            'decks_remaining': self.shoe.decks_remaining,
            'dealer_blackjack': self._dealer_blackjack,
            'insurance_available': self.dealer_hand.cards[0] == 'A',
        }


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

ACTION_MAP = {
    'h': Action.HIT,
    's': Action.STAND,
    'd': Action.DOUBLE,
}

ACTION_LABELS = {
    Action.HIT:    '[h]it',
    Action.STAND:  '[s]tand',
    Action.DOUBLE: '[d]ouble',
}


def _prompt_action(legal: list[Action]) -> Action:
    labels = '  '.join(ACTION_LABELS[a] for a in legal)
    valid_keys = {k for k, v in ACTION_MAP.items() if v in legal}
    while True:
        raw = input(f"\nAction? {labels}: ").strip().lower()
        if raw in valid_keys:
            return ACTION_MAP[raw]
        print(f"  Invalid. Choose from: {', '.join(sorted(valid_keys))}")


def _prompt_insurance(bet: float) -> float:
    max_insurance = bet / 2
    while True:
        raw = input(f"  Insurance? (max ${max_insurance:.2f}, or 0 to skip): $").strip()
        try:
            amount = float(raw)
            if 0 <= amount <= max_insurance:
                return amount
            print(f"  Enter a value between $0 and ${max_insurance:.2f}")
        except ValueError:
            print("  Enter a number.")


def _prompt_bet(min_bet: float, max_bet: float) -> float:
    """Returns bet amount."""
    while True:
        raw = input(f"  Bet ${min_bet:.0f}-${max_bet:.0f} (or 0 to sit out): $").strip().lower()
        if not raw:
            continue
        try:
            bet = float(raw)
            if bet == 0 or (min_bet <= bet <= max_bet):
                return bet
            print(f"  Bet must be 0, or between ${min_bet:.0f} and ${max_bet:.0f}.")
        except ValueError:
            print("  Enter a number.")


def _print_state(info: dict, hide_dealer: bool = True):
    ph = info['player_hand']
    dh = info['dealer_hand']
    print(f"\n  Dealer: ", end='')
    if hide_dealer:
        print(f"{dh.cards[0]}  [?]")
    else:
        print(dh)
    print(f"  You:    {ph}{'  (doubled)' if ph.doubled else ''}")


def play_interactive(
    num_decks: int = 2,
    starting_balance: float = 0.0,
    min_bet: float = 10.0,
    max_bet: float = 1000.0,
):
    env = BlackjackEnv(num_decks=num_decks)
    balance = starting_balance
    walked_away = False

    print(f"\n{'='*48}")
    print(f"  BLACKJACK  —  {num_decks}-deck shoe")
    print(f"  Starting balance: ${balance:.2f}")
    print(f"  Bet limits: ${min_bet:.0f} – ${max_bet:.0f}")
    print(f"  Dealer hits soft 17  |  Blackjack pays 3:2")
    print(f"  Play until shoe is exhausted. Bet 0 to sit out.")
    print(f"{'='*48}")

    while not env.shoe.needs_reshuffle():
        print(f"\n{'─'*48}")
        try:
            bet = _prompt_bet(min_bet, max_bet)
        except (EOFError, KeyboardInterrupt):
            walked_away = True
            break

        _, info = env.reset(bet=bet)
        _print_state(info, hide_dealer=True)

        # Insurance when dealer shows Ace (offered before revealing hole card)
        insurance_bet = 0.0
        if info['insurance_available']:
            insurance_bet = _prompt_insurance(bet)

        # Dealer blackjack — hand ends before player acts
        if info['dealer_blackjack']:
            print("\n  *** DEALER BLACKJACK ***")
            _print_state(info, hide_dealer=False)
            reward = env._dealer_play()  # scores main bet (dealer loop is a no-op at 21)
            insurance_profit = 2 * insurance_bet  # pays 2:1 on insurance bet
            if insurance_bet > 0:
                print(f"  Insurance pays: +${insurance_profit:.2f}")
            balance += _resolve(reward) + insurance_profit
            continue

        # Insurance loses — dealer confirmed no blackjack
        if insurance_bet > 0:
            balance -= insurance_bet
            print(f"  Insurance lost: -${insurance_bet:.2f}")

        # Player natural blackjack (dealer confirmed no BJ)
        if info['player_hand'].is_blackjack:
            print("\n  *** BLACKJACK! ***")
            reward = env._dealer_play()
            _print_state(env._info(), hide_dealer=False)
            balance += _resolve(reward)
            continue

        # Normal player turn
        terminated = False
        reward = 0.0
        while not terminated:
            if bet == 0:
                # Print once and stand automatically since we sit out
                if len(info['player_hand'].cards) == 2:
                    print("  (Sitting out this hand)")
                _, reward, terminated, _, info = env.step(Action.STAND)
                _print_state(info, hide_dealer=not terminated)
                continue

            legal = env.legal_actions()
            action = _prompt_action(legal)
            _, reward, terminated, _, info = env.step(action)
            _print_state(info, hide_dealer=not terminated)

        balance += _resolve(reward)

    reason = "Walked away." if walked_away else "Shoe exhausted."
    print(f"\n{'='*48}")
    print(f"  {reason}")
    print(f"  Final balance: ${balance:.2f}")
    print(f"{'='*48}\n")


def _resolve(reward: float) -> float:
    """Print outcome and return balance delta."""
    if reward > 0:
        print(f"\n  >> WIN  +${reward:.2f}")
    elif reward < 0:
        print(f"\n  >> LOSE  -${abs(reward):.2f}")
    else:
        print(f"\n  >> PUSH  ${0:.2f}")
    return reward


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Blackjack simulator')
    parser.add_argument('--decks', type=int, default=2, help='Number of decks (default: 2)')
    parser.add_argument('--balance', type=float, default=0.0, help='Starting balance (default: 0)')
    parser.add_argument('--min-bet', type=float, default=10.0, help='Minimum bet (default: 10)')
    parser.add_argument('--max-bet', type=float, default=1000.0, help='Maximum bet (default: 1000)')
    parser.add_argument('--penetration', type=float, default=0.75,
                        help='Shoe penetration before reshuffle (default: 0.75)')
    args = parser.parse_args()

    play_interactive(
        num_decks=args.decks,
        starting_balance=args.balance,
        min_bet=args.min_bet,
        max_bet=args.max_bet,
    )
