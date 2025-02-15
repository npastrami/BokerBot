'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import math
import random

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from training.neural_net import DeepCFRModel




class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''

        #Set up dictionary to convert rank to integer
        self.rank_to_int = dict()
        for i in range(2,10):
            self.rank_to_int[str(i)] = i

        for i, j in enumerate(["T", "J", "Q", "K"]):
            self.rank_to_int[j] = i+10

        self.rank_to_int["A"] = 1


        #Set up dictionary to convert suit to integer
        self.suit_to_int = dict()

        for i, j in enumerate(["d", "c", "h", "s"]):
            self.suit_to_int[j] = i

        self.nbets = 10
        self.nactions = 5

        #NOTE: I believe the number of cards types should be four cus [hole, flop, [turn, river]]
        #but it might indeed be 4

        self.model = DeepCFRModel(2, self.nbets, self.nactions)

        self.idx_to_action = dict()
        
        for idx, action in enumerate(["Fold", "Check", "Call", "Raise 1/2", "Raise 3/2"]):
            self.idx_to_action[idx] = action

        pass

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''

        self.bets = []
        #my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[active]  # your cards
        #big_blind = bool(active)  # True if you are the big blind
        #my_bounty = round_state.bounties[active]  # your current bounty rank
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        #my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        #street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        #opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        
        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        bounty_rank = previous_state.bounties[active]  # your bounty rank

        # The following is a demonstration of accessing illegal information (will not work)
        opponent_bounty_rank = previous_state.bounties[1-active]  # attempting to grab opponent's bounty rank

        if my_bounty_hit:
            print("I hit my bounty of " + bounty_rank + "!")
        if opponent_bounty_hit:
            print("Opponent hit their bounty of " + opponent_bounty_rank + "!")
    
    def get_card_num(self, card):

        card_rank = self.rank_to_int[card[0]]
        card_suit = self.suit_to_int[card[1]]

        card_num = (card_rank-1) + 13*(card_suit)

        return t.tensor([card_num])

    def tensorize_cards(self, my_hand, board):

        hole_tensor = t.tensor([self.get_card_num(card) for card in my_hand]).unsqueeze(0)

        board_nums = [self.get_card_num(card) for card in board]

        while len(board_nums) < 5:
            board_nums.append(-1)

        flop_tensor = t.tensor(board_nums[:3]).unsqueeze(0)
        turn_tensor = t.tensor(board_nums[3]).unsqueeze(0)
        river_tensor = t.tensor(board_nums[4]).unsqueeze(0)

        return [hole_tensor, flop_tensor, turn_tensor, river_tensor]

    def tensorize_bets(self):

        last_bets = self.bets[:][-self.nbets:]

        while len(last_bets) < self.nbets:
            last_bets.append(-1)

        tensorized_bets = t.tensor(last_bets, dtype=t.float32).unsqueeze(0)

        return tensorized_bets
    
    
    def tensorize_mask(self, legal_actions):

        mask_tensor = t.zeros(self.nactions, dtype=t.float32)

        if FoldAction in legal_actions:
            mask_tensor[0] = 1
        
        if CheckAction in legal_actions:
            mask_tensor[1] = 1
        
        if CallAction in legal_actions:
            mask_tensor[2] = 1
        
        if RaiseAction in legal_actions:

            if self.min_raise <= math.ceil(self.pot*1/2) <= self.max_raise:
                mask_tensor[3] = 1
            if self.min_raise <= math.ceil(self.pot*3/2) <= self.max_raise:
                mask_tensor[4] = 1
        
        return mask_tensor


    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        self.bets.append(continue_cost)
        self.pot = my_contribution + opp_contribution
        self.my_stack = my_stack

        print("The bets are: ", self.bets)

        if RaiseAction in legal_actions:
           min_raise, max_raise = round_state.raise_bounds() # the smallest and largest numbers of chips for a legal bet/raise
           self.min_raise = min_raise
           self.max_raise = max_raise 
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip  # the cost of a maximum bet/raise


        tensorized_bets = self.tensorize_bets()
        tensorized_cards = self.tensorize_cards(my_cards, board_cards)
        mask_tensor = self.tensorize_mask(legal_actions)

        model_regrets = self.model(tensorized_cards, tensorized_bets).squeeze(0)
        model_regrets = F.relu(mask_tensor*model_regrets)

        if t.sum(model_regrets) < 0.001:
            model_regrets = mask_tensor*t.ones(self.nactions)
        
        action_probabilities = model_regrets/(t.sum(model_regrets))

        selected_idx = int(t.multinomial(action_probabilities, 1, replacement=True))
        selected_action = self.idx_to_action[selected_idx]

        if selected_action == "Fold":
            output = FoldAction()

        elif selected_action == "Check":
            output = CheckAction()
        
        elif selected_action == "Call":
            output = CallAction()
        
        elif selected_action == "Raise 1/2":
            half_pot = math.ceil(self.pot*1/2)
            output = RaiseAction(half_pot)
        
        elif selected_action == "Raise 3/2":
            three_half_pot = math.ceil(self.pot*3/2)
            output = RaiseAction(three_half_pot)

        if isinstance(output, RaiseAction):
            self.bets.append(output.amount - my_pip)
        else:
            self.bets.append(0)

        return output


if __name__ == '__main__':
    run_bot(Player(), parse_args())
