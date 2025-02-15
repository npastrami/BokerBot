from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7
import numpy as np
from collections import defaultdict

class PublicBelief:
    """Enhanced public belief state combining ReBeL and Q-learning"""
    def __init__(self, street, board_cards, pot, active_player):
        self.street = street
        self.board_cards = board_cards
        self.pot = pot
        self.active_player = active_player
        self.value = 0  # ReBeL value
        self.q_values = defaultdict(float)  # Q-learning values for each action
        self.regret_sum = np.zeros(3)  # MCCFR regrets
        self.policy = np.zeros(3)  # Current policy

class HybridPokerBot(Bot):
    def __init__(self):
        super().__init__()
        ### same exact mccft setup
        # Combined state tracking
        self.belief_states = {}  # Maps info_sets to PublicBelief objects
        self.regret_sum = defaultdict(lambda: np.zeros(3))
        self.strategy_sum = defaultdict(lambda: np.zeros(3))
        
        # Q-learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
        # MCCFR parameters
        self.min_regret = -300
        self.explore_prob = 0.1
        
        # Game state tracking
        self.my_cards = None
        self.big_blind = None
        self.last_state = None
        self.last_action = None
        
    def get_public_belief(self, round_state, active):
        """Get or create public belief state"""
        info_set = self._create_info_set(round_state, active)
        
        if info_set not in self.belief_states:
            street = round_state.street
            board_cards = round_state.deck[:street] if street > 0 else []
            pot = sum(STARTING_STACK - stack for stack in round_state.stacks)
            self.belief_states[info_set] = PublicBelief(street, board_cards, pot, active)
            
        return self.belief_states[info_set]
    # get info set in rlrebel
    def _create_info_set(self, round_state, active):
        """Create unique string key for current game state"""
        street = round_state.street
        cards = sorted([str(card) for card in round_state.hands[active]])
        
        if street == 0:
            return f"preflop:{cards}:{self.big_blind}"
            
        board = round_state.deck[:street] if street > 0 else []
        board_str = ','.join(sorted([str(card) for card in board]))
        pot = sum(STARTING_STACK - stack for stack in round_state.stacks)
        
        return f"{street}:{cards}:{board_str}:{pot}:{self.big_blind}"
    # get_strategy in rlrebel    
    def get_mccfr_strategy(self, belief_state):
        """Get regret-matched strategy"""
        regrets = np.maximum(belief_state.regret_sum, self.min_regret)
        regrets = np.maximum(regrets, 0)
        
        total = np.sum(regrets)
        if total > 0:
            return regrets / total
        return np.ones(3) / 3
        
    def get_q_strategy(self, belief_state, legal_actions):
        """Get Q-learning based strategy"""
        if random.random() < self.epsilon:
            return np.ones(3) / 3  # Exploration: uniform random
            
        # Convert Q-values to probabilities using softmax
        q_values = np.zeros(3)
        for i, action_type in enumerate(['fold/check', 'call', 'raise']):
            q_values[i] = belief_state.q_values[action_type]
            
        exp_q = np.exp(q_values)
        return exp_q / np.sum(exp_q)
        
    def update_q_value(self, belief_state, action_idx, reward, next_belief_state=None):
        """Update Q-values using Q-learning"""
        action_type = ['fold/check', 'call', 'raise'][action_idx]
        current_q = belief_state.q_values[action_type]
        
        if next_belief_state:
            next_max_q = max(next_belief_state.q_values.values())
            target = reward + self.gamma * next_max_q
        else:
            target = reward
            
        # Update Q-value with regret information
        regret = belief_state.regret_sum[action_idx]
        regret_weight = 0.5  # Weight for combining Q-learning with regret
        
        new_q = current_q + self.alpha * (target - current_q)
        belief_state.q_values[action_type] = (1 - regret_weight) * new_q + regret_weight * regret
        
    def get_action(self, game_state, round_state, active):
        """Main action selection method combining MCCFR, ReBeL, and Q-learning"""
        legal_actions = round_state.legal_actions()
        belief_state = self.get_public_belief(round_state, active)
        
        # Calculate hand strength
        community = round_state.deck[:round_state.street] if round_state.street > 0 else []
        strength = self.calc_strength(self.my_cards, 100, community)
        
        # Get strategies from both approaches
        mccfr_strategy = self.get_mccfr_strategy(belief_state)
        q_strategy = self.get_q_strategy(belief_state, legal_actions)
        
        # Combine strategies
        combined_strategy = 0.7 * mccfr_strategy + 0.3 * q_strategy
        
        # Adjust based on hand strength
        if round_state.street == 0:  # Preflop
            if strength < 0.5:
                combined_strategy[2] = 0  # Never raise
                combined_strategy[0] *= 2  # Prefer folding
            elif strength > 0.8:
                combined_strategy[2] *= 1.2
        else:  # Postflop
            if strength < 0.6:
                combined_strategy[2] = 0
                combined_strategy[0] *= 1.5
            elif strength > 0.85:
                combined_strategy[2] *= 1.2
                
        # If out of position, be more conservative
        if not self.big_blind:
            combined_strategy[2] *= 0.7
            
        # Normalize
        combined_strategy = np.maximum(combined_strategy, 0)
        total = np.sum(combined_strategy)
        if total > 0:
            combined_strategy = combined_strategy / total
        else:
            combined_strategy = np.array([0.8, 0.15, 0.05])
            
        action = self.get_action_from_strategy(combined_strategy, legal_actions, round_state, active, strength)
        
        # Store state and action for Q-learning update
        self.last_state = belief_state
        self.last_action = self.get_action_index(action)
        
        return action
        
    def get_action_index(self, action):
        """Convert action to index"""
        if isinstance(action, (FoldAction, CheckAction)):
            return 0
        elif isinstance(action, CallAction):
            return 1
        elif isinstance(action, RaiseAction):
            return 2
        return 0
        
    def handle_round_over(self, game_state, terminal_state, active):
        """Update both MCCFR regrets and Q-values"""
        if self.last_state and self.last_action is not None:
            my_delta = terminal_state.deltas[active]
            
            # Update Q-value
            self.update_q_value(self.last_state, self.last_action, my_delta)
            
            # Update MCCFR regrets
            realized_values = np.zeros(3)
            if my_delta > 0:
                if self.last_action == 2:  # Raise was good
                    realized_values[2] = my_delta
                else:  # Call was good
                    realized_values[1] = my_delta
            elif my_delta < 0:
                realized_values[0] = abs(my_delta)  # Should have folded
                
            # Update regrets
            strategy = self.get_mccfr_strategy(self.last_state)
            node_value = np.dot(strategy, realized_values)
            
            for action in range(3):
                regret = realized_values[action] - node_value
                self.last_state.regret_sum[action] += regret
            
        self.last_state = None
        self.last_action = None
    # same exact as rlrebel               
    def get_action_from_strategy(self, strategy, legal_actions, round_state, active, strength):
        """Convert strategy probabilities into actual poker actions"""
        sorted_actions = np.argsort(strategy)[::-1]
        
        for action_idx in sorted_actions:
            if action_idx == 0:
                if FoldAction in legal_actions:
                    return FoldAction()
                elif CheckAction in legal_actions:
                    return CheckAction()
                    
            elif action_idx == 1 and CallAction in legal_actions:
                return CallAction()
                
            elif action_idx == 2:
                try:
                    if RaiseAction in legal_actions and strength >= 0.5:
                        min_raise, max_raise = round_state.raise_bounds()
                        my_pip = round_state.pips[active]
                        my_stack = round_state.stacks[active]
                        opp_pip = round_state.pips[1-active]
                        
                        pot_total = my_pip + opp_pip
                        continue_cost = opp_pip - my_pip

                        # More conservative raise sizing
                        if round_state.street == 0:  # Preflop
                            raise_amount = min_raise + int(continue_cost)
                        else:  # Postflop
                            raise_amount = min_raise + int(continue_cost + (pot_total * 0.5))
                        
                        # Ensure raise is valid
                        raise_amount = max(min_raise, raise_amount)
                        raise_amount = min(max_raise, raise_amount)
                        raise_cost = raise_amount - my_pip
                        
                        if raise_cost <= my_stack and raise_amount > my_pip:
                            return RaiseAction(raise_amount)
                except Exception:
                    pass
                    
        # Fallback actions
        if CheckAction in legal_actions:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()
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
        self.last_state = None
        self.last_action = None

if __name__ == '__main__':
    run_bot(HybridPokerBot(), parse_args())