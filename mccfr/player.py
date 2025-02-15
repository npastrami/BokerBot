from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7
import numpy as np
from collections import defaultdict

class MCCFRBot(Bot):
    def __init__(self):
        super().__init__()
        
        ### same exact mccft setup
        # MCCFR state tracking
        self.regret_sum = defaultdict(lambda: np.zeros(3))  # 3 actions: fold/check, call, raise
        self.strategy_sum = defaultdict(lambda: np.zeros(3))
        self.iterations = 0
        
        # Game state tracking
        self.my_cards = None
        self.big_blind = None
        
        # Action mapping
        self.action_map = {
            0: (FoldAction, CheckAction),  # fold if available, otherwise check
            1: CallAction,
            2: RaiseAction
        }
        
        # Hyperparameters
        self.explore_prob = 0.1
        self.min_regret = -300
        
        self.round_counter = 0  
    
    # create info set in hybrid 
    def get_info_set(self, round_state, active):
        """Create unique string key for current game state"""
        street = round_state.street
        cards = sorted([str(card) for card in round_state.hands[active]])
        
        if street == 0:
            return f"preflop:{cards}:{self.big_blind}"
            
        board = round_state.deck[:street] if street > 0 else []
        board_str = ','.join(sorted([str(card) for card in board]))
        
        pot = sum(STARTING_STACK - stack for stack in round_state.stacks)
        return f"{street}:{cards}:{board_str}:{pot}:{self.big_blind}"
    
    # get_mccfr_strategy in hybrid 
    def get_strategy(self, info_set):
        """Get current strategy for an information set using regret matching"""
        regrets = np.maximum(self.regret_sum[info_set], self.min_regret)
        regrets = np.maximum(regrets, 0)
        
        norm_sum = np.sum(regrets)
        if norm_sum > 0:
            return regrets / norm_sum
        return np.ones(3) / 3  # Equal probability if no regrets
        
    def update_strategy(self, info_set, strategy, realized_value, node_value):
        """Update regrets and strategy sums"""
        for action in range(3):
            regret = realized_value[action] - node_value
            self.regret_sum[info_set][action] += regret
            self.strategy_sum[info_set][action] += strategy[action]
            
            
    # same exact as hybrid        
    def get_action_from_strategy(self, strategy, legal_actions, round_state, active, strength):
        """Convert strategy probabilities into actual poker actions with hand strength thresholds."""
        sorted_actions = np.argsort(strategy)[::-1]
        legal_action_types = set(legal_actions)
        
        self.round_counter += 1  # Increment round counter
        print(f"\n=== DEBUG: Round #{self.round_counter} ===")
        print("length of legal actions:", len(legal_actions))
        print("=== DEBUG: get_action_from_strategy ===")
        print(f"DEBUG: Strategy probabilities: {strategy}")
        print(f"DEBUG: Legal actions: {[action.__name__ for action in legal_action_types]}")
        print(f"DEBUG: Current street: {round_state.street}, Active player: {active}, Strength: {strength}")

        # Define thresholds for raising and calling based on hand strength
        raise_threshold = 0.6  # Only raise if strength > 0.6
        call_threshold = 0.4   # Only call if strength > 0.4

        # Determine continue cost
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip

        # Handle situations with a pending cost
        if continue_cost > 0:
            if CallAction in legal_action_types:
                if strength > call_threshold:
                    print(f"DEBUG: Selected CallAction (continue cost: {continue_cost}, strength: {strength})")
                    return CallAction()
                else:
                    print("DEBUG: Skipping CallAction (strength too low)")
                    return FoldAction()
            else:
                print("DEBUG: No CallAction available, folding")
                return FoldAction()

        # Handle normal actions when no continue cost
        for action_idx in sorted_actions:
            if action_idx == 0:  # Fold/Check
                if CheckAction in legal_action_types and continue_cost == 0:
                    print("DEBUG: Selected CheckAction (no pending bet)")
                    return CheckAction()
                elif FoldAction in legal_action_types:
                    print("DEBUG: Selected FoldAction")
                    return FoldAction()

            elif action_idx == 2 and RaiseAction in legal_action_types:  # Raise
                if strength > raise_threshold:  # Only raise if hand strength is high enough
                    try:
                        min_raise, max_raise = round_state.raise_bounds()
                        pot_total = my_pip + opp_pip

                        # Ensure the raise amount is always within valid bounds
                        if round_state.street < 3:  # Preflop & Flop
                            raise_amount = max(min_raise, int(my_pip + continue_cost + 0.4 * (pot_total + continue_cost)))
                        else:  # Turn & River
                            raise_amount = max(min_raise, int(my_pip + continue_cost + 0.75 * (pot_total + continue_cost)))
                        
                        raise_amount = min(max_raise, raise_amount) + 1 # Ensure raise is not above max

                        raise_cost = raise_amount - my_pip
                        if raise_cost <= round_state.stacks[active]:
                            print(f"DEBUG: Selected RaiseAction with amount={raise_amount}")
                            return RaiseAction(raise_amount)
                    except Exception as e:
                        print(f"ERROR: Exception during RaiseAction computation: {e}")
                        pass
                else:
                    print("DEBUG: Skipping RaiseAction (strength too low)")

        # Fallback actions if no preferred action is valid
        if CheckAction in legal_action_types and continue_cost == 0:
            print("DEBUG: Fallback -> CheckAction (no pending bet)")
            return CheckAction()
        if CallAction in legal_action_types:
            print("DEBUG: Fallback -> CallAction")
            return CallAction()
        print("DEBUG: Fallback -> FoldAction")
        return FoldAction()

    
    # calc strength the same    
    def calc_strength(self, hole, iterations, community=[]):
        """Monte Carlo hand strength calculation"""
        deck = eval7.Deck()
        hole_cards = [eval7.Card(card) for card in hole]
        
        if community:
            community_cards = [eval7.Card(card) for card in community]
            for card in community_cards:
                deck.cards.remove(card)
                
        for card in hole_cards:
            deck.cards.remove(card)
            
        score = 0
        for _ in range(iterations):
            deck.shuffle()
            
            remaining = 5 - len(community)
            draw = deck.peek(remaining + 2)
            opp_hole = draw[:2]
            extra_community = draw[2:]
            
            our_hand = hole_cards + (community_cards if community else []) + extra_community
            opp_hand = opp_hole + (community_cards if community else []) + extra_community
            
            our_value = eval7.evaluate(our_hand)
            opp_value = eval7.evaluate(opp_hand)
            
            score += 2 if our_value > opp_value else (1 if our_value == opp_value else 0)
            
        return score / (2 * iterations)
        
    def handle_new_round(self, game_state, round_state, active):
        """Initialize tracking for new round"""
        self.my_cards = round_state.hands[active]
        self.big_blind = bool(active)
        
    def get_action(self, game_state, round_state, active):
        """Main action selection method"""
        legal_actions = round_state.legal_actions()
        
        # Get current info set and strategy
        info_set = self.get_info_set(round_state, active)
        strategy = self.get_strategy(info_set)
        
        # Calculate hand strength
        community = round_state.deck[:round_state.street] if round_state.street > 0 else []
        strength = self.calc_strength(self.my_cards, 100, community)
        
        # Adjust strategy based on hand strength
        if strength > 0.8:  # Very strong hands
            strategy[2] *= 1.5  # Increase raise probability
        elif strength < 0.3:  # Weak hands
            strategy[0] *= 1.5  # Increase fold/check probability
            
        # Normalize strategy after adjustments
        strategy = strategy / np.sum(strategy)
        
        # Get actual action
        action = self.get_action_from_strategy(strategy, legal_actions, round_state, active, strength)
        print("strategy", strategy)
        # Update strategy tracking
        self.iterations += 1
        print("action", action)
        return action

    def handle_round_over(self, game_state, terminal_state, active):
        """Update strategy based on round outcome"""
        my_delta = terminal_state.deltas[active]
        previous_state = terminal_state.previous_state

        if previous_state:
            info_set = self.get_info_set(previous_state, active)
            strategy = self.get_strategy(info_set)
            
            realized_values = np.zeros(3)  # fold/check, call, raise
            
            if my_delta > 0:  # Won the round
                if previous_state.button == active:  # Was our action
                    if any(isinstance(a, RaiseAction) for a in previous_state.legal_actions()):
                        realized_values[2] = my_delta  # Raise was good
                    else:
                        realized_values[1] = my_delta  # Call was good
            elif my_delta < 0:  # Lost the round
                if previous_state.button == active:
                    realized_values[0] = abs(my_delta)  # Should have folded
            
            node_value = np.dot(strategy, realized_values)
            self.update_strategy(info_set, strategy, realized_values, node_value)

if __name__ == '__main__':
    run_bot(MCCFRBot(), parse_args())