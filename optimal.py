import sys
import argparse
from sim import BlackjackEnv, Action

def get_action(player_sum, dealer_upcard, usable_ace, can_double, true_count):
    """
    Returns optimal action using H17 basic strategy and Illustrious 18 deviations.
    """
    if not usable_ace:
        # Illustrious 18 Deviations
        if player_sum == 16 and dealer_upcard == 10 and true_count >= 0: return Action.STAND
        if player_sum == 15 and dealer_upcard == 10 and true_count >= 4: return Action.STAND
        if player_sum == 14 and dealer_upcard == 10 and true_count >= 3: return Action.STAND
        if player_sum == 12 and dealer_upcard == 3 and true_count >= 2: return Action.STAND
        if player_sum == 12 and dealer_upcard == 2 and true_count >= 3: return Action.STAND
        if player_sum == 16 and dealer_upcard == 9 and true_count >= 5: return Action.STAND
        if player_sum == 13 and dealer_upcard == 2 and true_count <= -1: return Action.HIT
        if player_sum == 12 and dealer_upcard == 4 and true_count <= 0: return Action.HIT
        if player_sum == 12 and dealer_upcard == 5 and true_count <= -2: return Action.HIT
        if player_sum == 12 and dealer_upcard == 6 and true_count <= -1: return Action.HIT
        
        if can_double:
            if player_sum == 10 and dealer_upcard == 10 and true_count >= 4: return Action.DOUBLE
            if player_sum == 10 and dealer_upcard == 11 and true_count >= 4: return Action.DOUBLE
            if player_sum == 9 and dealer_upcard == 2 and true_count >= 1: return Action.DOUBLE
            if player_sum == 9 and dealer_upcard == 7 and true_count >= 3: return Action.DOUBLE

    # Basic Strategy (H17)
    if not usable_ace:
        if player_sum <= 8:
            return Action.HIT
        elif player_sum == 9:
            return Action.DOUBLE if can_double and 3 <= dealer_upcard <= 6 else Action.HIT
        elif player_sum == 10:
            return Action.DOUBLE if can_double and 2 <= dealer_upcard <= 9 else Action.HIT
        elif player_sum == 11:
            return Action.DOUBLE if can_double and 2 <= dealer_upcard <= 11 else Action.HIT
        elif player_sum == 12:
            return Action.STAND if 4 <= dealer_upcard <= 6 else Action.HIT
        elif 13 <= player_sum <= 16:
            return Action.STAND if 2 <= dealer_upcard <= 6 else Action.HIT
        else:
            return Action.STAND
    else:
        # Soft totals
        if player_sum <= 14:
            return Action.DOUBLE if can_double and 5 <= dealer_upcard <= 6 else Action.HIT
        elif player_sum <= 16:
            return Action.DOUBLE if can_double and 4 <= dealer_upcard <= 6 else Action.HIT
        elif player_sum == 17:
            return Action.DOUBLE if can_double and 2 <= dealer_upcard <= 6 else Action.HIT
        elif player_sum == 18:
            if can_double and 2 <= dealer_upcard <= 6:
                return Action.DOUBLE
            elif 2 <= dealer_upcard <= 8:
                return Action.STAND
            else:
                return Action.HIT
        elif player_sum == 19:
            if can_double and dealer_upcard == 6:
                return Action.DOUBLE
            else:
                return Action.STAND
        else:
            return Action.STAND

def get_bet(true_count, min_bet=10, max_bet=1000):
    """
    Bet sizing based on True Count.
    Edge typically turns positive at TC >= +1.
    We sit out (bet 0) when the true count is neutral or negative.
    """
    if true_count <= 2.0:
        return 0.0
    
    # Scale bet purely based on advantage, clamped to max_bet
    bet = min_bet + (true_count - 2.0) * 200
    return float(max(min_bet, min(max_bet, bet)))

def play_agent(shoes=100, decks=6, penetration=0.75):
    balance = 0.0
    total_wagered = 0.0
    hands_played = 0

    print(f"Starting simulation of {shoes} shoes with optimal player...")
    print("Agent is using True Count exactly, H17 Basic Strategy with Illustrious 18 deviations, and spread-betting.")
    print("-" * 60)
    
    for i in range(shoes):
        env = BlackjackEnv(num_decks=decks, penetration=penetration)
        
        while not env.shoe.needs_reshuffle():
            tc = env.shoe.true_count
            bet = get_bet(tc)
            
            obs, info = env.reset(bet=bet)
            total_wagered += bet
            hands_played += 1
            
            # 1. Insurance (Illustrious 18)
            insurance_bet = 0.0
            if info['insurance_available']:
                if tc >= 3.0:
                    insurance_bet = bet / 2.0
                    total_wagered += insurance_bet
                    
            # 2. Dealer blackjack ends the round
            if info['dealer_blackjack']:
                reward = env._dealer_play()
                insurance_profit = 2 * insurance_bet # pays 2:1 -> net profit is 2x the insurance bet
                balance += reward + insurance_profit
                continue
                
            # If dealer didn't have BJ, insurance is lost
            if insurance_bet > 0:
                balance -= insurance_bet
                
            # 3. Player natural
            if info['player_hand'].is_blackjack:
                reward = env._dealer_play()
                balance += reward
                continue
                
            # 4. Standard play
            terminated = False
            reward = 0.0
            
            while not terminated:
                player_sum, upcard, usable_ace = obs
                legal = env.legal_actions()
                can_double = Action.DOUBLE in legal
                
                action = get_action(player_sum, upcard, usable_ace, can_double, tc)
                
                if action == Action.DOUBLE and not can_double:
                    action = Action.HIT
                    
                if action == Action.DOUBLE:
                    total_wagered += bet # Extra wagered amount
                    
                obs, reward, terminated, _, info = env.step(action)
                
            balance += reward
            
        if (i + 1) % 100 == 0 or (i + 1) == shoes:
            roi = (balance / total_wagered * 100) if total_wagered else 0
            print(f"Shoe {i + 1:4d} | Balance: ${balance:9.2f} | Wagered: ${total_wagered:10.2f} | Edge (ROI): {roi:5.2f}%")

    print("-" * 60)
    print("Simulation Complete!")
    print(f"Total Hands Played : {hands_played}")
    print(f"Total Amount Wagered: ${total_wagered:.2f}")
    print(f"Final Balance      : ${balance:.2f}")
    roi = (balance / total_wagered * 100) if total_wagered > 0 else 0
    print(f"Realized Edge (ROI): {roi:.3f}%")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Optimal Blackjack Agent")
    parser.add_argument("--shoes", type=int, default=1000, help="Number of shoes to play")
    parser.add_argument("--decks", type=int, default=6, help="Number of decks in shoe")
    parser.add_argument("--penetration", type=float, default=0.75, help="Shoe penetration")
    args = parser.parse_args()
    
    play_agent(shoes=args.shoes, decks=args.decks, penetration=args.penetration)
