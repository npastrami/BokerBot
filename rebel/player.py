import torch
import torch.nn as nn
import torch.optim as optim
import random
import eval7
import numpy as np
from collections import defaultdict

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class PublicBelief:
    """Represents a public belief state in the game"""
    def __init__(self, street, board_cards, pot, active_player):
        self.street = street
        self.board_cards = board_cards
        self.pot = pot
        self.active_player = active_player
        self.value = 0
        self.policy = defaultdict(float)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class CFRPlusPolicy(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        # CFR+ specific attributes
        self.regret_sum = defaultdict(lambda: torch.zeros(num_actions))
        self.strategy_sum = defaultdict(lambda: torch.zeros(num_actions))
        self.reach_probs = defaultdict(float)
        
    def forward(self, x):
        logits = self.network(x)
        return logits
        
    def get_strategy(self, state_key):
        """Get current strategy through regret matching"""
        regrets = self.regret_sum[state_key]
        positive_regrets = torch.clamp(regrets, min=0)
        sum_positive_regret = positive_regrets.sum()
        
        if sum_positive_regret > 0:
            strategy = positive_regrets / sum_positive_regret
        else:
            strategy = torch.ones_like(regrets) / len(regrets)
        
        return strategy
    
    def update_regrets(self, state_key, regrets, reach_prob, iteration):
        """CFR+ regret update with regret matching"""
        # CFR+ uses positive regrets only
        self.regret_sum[state_key] = torch.clamp(
            self.regret_sum[state_key] + reach_prob * regrets,
            min=0
        )
        
        # Update strategy sum weighted by iteration (CFR+ specific)
        strategy = self.get_strategy(state_key)
        self.strategy_sum[state_key] += iteration * reach_prob * strategy
        
    def get_average_strategy(self, state_key):
        """Get average strategy across all iterations"""
        strategy_sum = self.strategy_sum[state_key]
        total = strategy_sum.sum()
        
        if total > 0:
            return strategy_sum / total
        return torch.ones_like(strategy_sum) / len(strategy_sum)

class NeuralReBeL(Bot):
    def __init__(self):
        super().__init__()
        self.input_dim = 10  # Adjust based on your state encoding
        self.num_actions = 4  # fold, call, check, raise
        
        # Neural Networks
        self.value_net = ValueNetwork(self.input_dim)
        self.policy_net = CFRPlusPolicy(self.input_dim, self.num_actions)
        
        # Optimizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.001)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Training parameters
        self.current_iteration = 0
        self.t_warm = 1000
        self.training_iters = 10000
        self.epsilon = 0.25
        self.discount = 0.99

    def handle_new_round(self, game_state, round_state, active):
        """Called when a new round starts. Called NUM_ROUNDS times."""
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

    def encode_state(self, public_belief, private_cards=None):
        """Convert game state to neural network input tensor"""
        street = torch.tensor([public_belief.street], dtype=torch.float32)
        pot = torch.tensor([public_belief.pot], dtype=torch.float32)
        active = torch.tensor([public_belief.active_player], dtype=torch.float32)
        
        # Encode board cards (one-hot or embedding)
        board_encoding = torch.zeros(5)  # Assuming max 5 community cards
        for i, card in enumerate(public_belief.board_cards):
            if i < len(board_encoding):
                board_encoding[i] = 1  # Simplified encoding, improve as needed
        
        # Encode private cards if available
        hand_encoding = torch.zeros(2)  # For two hole cards
        if private_cards:
            hand_encoding = torch.ones(2)  # Simplified encoding
            
        # Combine all features
        state = torch.cat([street, pot, active, board_encoding, hand_encoding])
        return state.unsqueeze(0)  # Add batch dimension

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
        """Calculate poker hand strength through Monte Carlo simulation"""
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

    def cfr_update(self, public_belief, action_utilities):
        """Perform CFR+ update for the current state"""
        state = self.encode_state(public_belief)
        state_key = (public_belief.street, tuple(public_belief.board_cards), 
                    public_belief.pot, public_belief.active_player)
        
        # Get current strategy and compute regrets
        strategy = self.policy_net.get_strategy(state_key)
        expected_value = torch.sum(action_utilities * strategy)
        regrets = action_utilities - expected_value
        
        # Update regrets and strategy (CFR+ style)
        self.policy_net.update_regrets(state_key, regrets, 
                                     self.policy_net.reach_probs[state_key],
                                     self.current_iteration)
        
        return expected_value

    def get_action(self, game_state, round_state, active):
        """Select action using neural networks with CFR+ and ε-greedy exploration"""
        legal_actions = round_state.legal_actions()
        public_belief = self.get_public_state(round_state, active)
        
        # Calculate hand strength
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:public_belief.street] if public_belief.street > 0 else []
        hand_strength = self.calc_hand_strength(my_cards, 100, board_cards)
        
        # Exploration phase
        if random.random() < self.epsilon:
            if hand_strength > 0.7 and RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(min_raise)
            elif hand_strength > 0.5 and CallAction in legal_actions:
                return CallAction()
            elif CheckAction in legal_actions:
                return CheckAction()
            else:
                return FoldAction()
        
        # Get CFR+ strategy
        state = self.encode_state(public_belief, my_cards)
        state_key = (public_belief.street, tuple(public_belief.board_cards), 
                    public_belief.pot, public_belief.active_player)
        strategy = self.policy_net.get_strategy(state_key)
        
        # Select action based on strategy probabilities and legal actions
        action_map = {0: FoldAction, 1: CallAction, 2: CheckAction, 3: RaiseAction}
        action_probs = strategy.detach()
        sorted_actions = torch.argsort(action_probs, descending=True)
        
        for action_idx in sorted_actions:
            action_class = action_map[action_idx.item()]
            if action_class in legal_actions:
                if action_class == RaiseAction:
                    min_raise, max_raise = round_state.raise_bounds()
                    return RaiseAction(min_raise)
                return action_class()
        
        return FoldAction()  # Fallback

    def handle_round_over(self, game_state, terminal_state, active):
        """Update networks with terminal state outcomes using CFR+"""
        my_delta = terminal_state.deltas[active]
        previous_state = terminal_state.previous_state
        public_belief = self.get_public_state(previous_state, active)
        
        # Update value network
        state = self.encode_state(public_belief)
        predicted_value = self.value_net(state)
        target_value = torch.tensor([[my_delta]], dtype=torch.float32)
        value_loss = nn.MSELoss()(predicted_value, target_value)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update iteration counter for CFR+
        self.current_iteration += 1

if __name__ == '__main__':
    run_bot(NeuralReBeL(), parse_args())