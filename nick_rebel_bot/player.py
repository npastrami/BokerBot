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
    """Represents a public belief state in the game"""
    def __init__(self, street, board_cards, pot, active_player):
        self.street = street
        self.board_cards = board_cards
        self.pot = pot
        self.active_player = active_player
        self.value = 0
        self.policy = defaultdict(float)

class ReBeL(Bot):
    def __init__(self):
        """
        Maps to initialization in REBEL-LINEAR-CFR-D
        - value_net corresponds to v in the algorithm
        - policy_net corresponds to π in the algorithm
        - t_warm = 1000 maps to twarm initialization
        - epsilon = 0.25 maps to ε in SAMPLELEAF function
        """
        super().__init__()
        self.value_net = defaultdict(float) # Value network v(r)
        self.policy_net = defaultdict(lambda: defaultdict(float)) # Policy network π
        self.t_warm = 1000 # twarm initialization, num of iters algo uses for basic policy generation
        self.training_iters = 10000 # T in the pseudocode
        self.epsilon = 0.25 # ε from SAMPLELEAF
        self.discount = 0.99 # discount factor, 0 to 1, causes bot to care almost equally about immediate and future rewards
        
    def handle_new_round(self, game_state, round_state, active):
        """
        Called when a new round starts. Called NUM_ROUNDS times.
        """
        self.my_bankroll = game_state.bankroll
        self.game_clock = game_state.game_clock
        self.round_num = game_state.round_num
        self.my_cards = round_state.hands[active]
        self.big_blind = bool(active)

    def get_public_state(self, round_state, active):
        """Convert RoundState to a public belief state"""
        street = round_state.street
        board_cards = round_state.deck[:street] if street > 0 else []
        pot = sum(STARTING_STACK - stack for stack in round_state.stacks)
        return PublicBelief(street, board_cards, pot, active)

    def update_value(self, public_belief, new_value):
        """
        Maps to value network update in REBEL-LINEAR-CFR-D
        - Updates v(r) based on observed outcomes
        - Uses exponential moving average for updates
        """
        key = (public_belief.street, tuple(public_belief.board_cards), 
               public_belief.pot, public_belief.active_player)
        self.value_net[key] = (1 - 0.1) * self.value_net[key] + 0.1 * new_value

    def update_policy(self, public_belief, action_type, prob):
        """
        Maps to UPDATEPOLICY in the algorithm
        - Updates π based on observed actions and outcomes
        - Uses exponential moving average for updates
        """
        key = (public_belief.street, tuple(public_belief.board_cards),
               public_belief.pot, public_belief.active_player)
        self.policy_net[key][action_type] = (1 - 0.1) * self.policy_net[key][action_type] + 0.1 * prob

    def get_action_type(self, action):
        """Get the type of an action instance"""
        if isinstance(action, FoldAction):
            return "fold"
        elif isinstance(action, CallAction):
            return "call"
        elif isinstance(action, CheckAction):
            return "check"
        elif isinstance(action, RaiseAction):
            return "raise"
        return None

    def create_action(self, action_type, round_state=None):
        """Create an action instance from type"""
        if action_type == "fold":
            return FoldAction()
        elif action_type == "call":
            return CallAction()
        elif action_type == "check":
            return CheckAction()
        elif action_type == "raise" and round_state is not None:
            min_raise, max_raise = round_state.raise_bounds()
            return RaiseAction(min_raise)
        return CheckAction()  # Default action

    def calc_hand_strength(self, hole, iters, community=[]):
        """
        Estimates poker hand strength through Monte Carlo simulation by sampling random opponent hands.
        Simulates 'iters' number of scenarios, comparing our hand against random opponent hands. 
        Args:
            hole (list): Our two hole cards (e.g. ["As", "Kh"])
            iters (int): Number of Monte Carlo iterations to run
            community (list): Currently visible community cards, empty for preflop (e.g. ["Jd", "Td", "4s"])
        Returns:
            float: Hand strength score between 0-1, where 1.0 = always winning, 0.5 = equal chances,
                    0.0 = always losing. Calculated as (wins + 0.5*ties)/iterations.
        """
        deck = eval7.Deck()
        hole_cards = [eval7.Card(card) for card in hole]
        
        if community:
            community_cards = [eval7.Card(card) for card in community]
            for card in community_cards:
                deck.cards.remove(card)
                
        for card in hole_cards:
            deck.cards.remove(card)
            
        score = 0
        for _ in range(iters):
            deck.shuffle()
            
            remaining_comm = 5 - len(community)
            draw = deck.peek(remaining_comm + 2)
            opp_hole = draw[:2]
            alt_community = draw[2:]
            
            our_hand = hole_cards + (community_cards if community else []) + alt_community
            opp_hand = opp_hole + (community_cards if community else []) + alt_community
            
            our_value = eval7.evaluate(our_hand)
            opp_value = eval7.evaluate(opp_hand)
            
            score += 2 if our_value > opp_value else (1 if our_value == opp_value else 0)
            
        return score / (2 * iters)

    def get_action(self, game_state, round_state, active):
        """
        Maps to SAMPLELEAF function in the algorithm
        - Implements the action selection logic based on:
          * Current PBS (public belief state)
          * Hand strength (part of COMPUTEEV)
          * Exploration (ε-greedy) from SAMPLELEAF
        """
        legal_actions = round_state.legal_actions()
        public_belief = self.get_public_state(round_state, active)
        
        # Calculate EV through hand strength (COMPUTEEV)
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:public_belief.street] if public_belief.street > 0 else []
        hand_strength = self.calc_hand_strength(my_cards, 100, board_cards)
        
        # Exploration (ε = 0.25) maps to SAMPLELEAF's uniform random action selection
        if random.random() < self.epsilon:  # "if i == i* and c < ε then"
            # Uniform random action selection based on hand strength thresholds
            if hand_strength > 0.7 and RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(min_raise)
            elif hand_strength > 0.5 and CallAction in legal_actions:
                return CallAction()
            elif CheckAction in legal_actions:
                return CheckAction()
            else:
                return FoldAction()
        
        # Exploitation maps to "sample an action ai according to πi(si(h))"
        if hand_strength > 0.8 and RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            raise_amount = min(max_raise, int(min_raise * 2.5))
            return RaiseAction(raise_amount)
        elif hand_strength > 0.6 and CallAction in legal_actions:
            return CallAction()
        elif CheckAction in legal_actions:
            return CheckAction()
        else:
            return FoldAction()

    def handle_round_over(self, game_state, terminal_state, active):
        """
        Maps to the update phase of REBEL-LINEAR-CFR-D
        - Updates value network based on terminal state outcomes
        - Adds to training data (Dv in algorithm)
        """
        my_delta = terminal_state.deltas[active]
        previous_state = terminal_state.previous_state
        
        # Update value network with actual reward
        public_belief = self.get_public_state(previous_state, active)
        self.update_value(public_belief, my_delta)

if __name__ == '__main__':
    run_bot(ReBeL(), parse_args())