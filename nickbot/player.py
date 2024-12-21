from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random 
import eval7

class Player(Bot):
    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.
        Arguments:
        Nothing.
        Returns:
        Nothing.
        '''
    def calc_strength(self, hole, iters, community = []):
        '''
        Using MC with iterations to evaluate hand strength
        Args:
        hole - our hold cards
        iters - number of times we run MC
        community - community cards
        '''
        
        deck = eval7.Deck() # deck of cards
        hole_cards = [eval7.Card(card) for card in hole] # our hole cards in eval7 friendly format
        
        # if community cards are not empty, we need to remove them from the deck
        # we don't want to draw them again against the MC
        if community != []:
            community_cards = [eval7.Card(card) for card in community]
            for card in community_cards:
                deck.cards.remove(card)
            
        for card in hole_cards:
            deck.cards.remove(card)
            
        # the score is the number of times we win, tie, or lose
        score = 0
        
        for _ in range(iters):
            deck.shuffle()
            
            # Lets see how many community cards we will need to draw
            if len(community) >= 5: # red river case (needs removal)
                #check the last communtiy card to see if it is red
                if community[-1][1] == 'h' or community[-1][1] == 'd':
                    _COMM = 1
                else:
                    _COMM = 0
            else:
                _COMM = 5 - len(community)
            
            _OPP = 2
            
            draw = deck.peek(_COMM + _OPP)
            opp_hole = draw[:_OPP]
            alt_community = draw[_OPP:]
            
            if community == []:
                our_hand = hole_cards + alt_community
                opp_hand = opp_hole + alt_community
            else:
                our_hand = hole_cards + community_cards + alt_community
                opp_hand = opp_hole + community_cards + alt_community
            
            our_hand_value = eval7.evaluate(our_hand)
            opp_hand_value = eval7.evaluate(opp_hand)

            if our_hand_value > opp_hand_value:
                score += 2 

            if our_hand_value == opp_hand_value:
                score += 1 
            else: 
                score += 0        

        hand_strength = score/(2*iters) # win probability 

        return hand_strength
    
    
    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.
        Arguments:
        game_state:
        round_state:
        active:
        Returns:
        Nothing
        '''
        my_bankroll = game_state.bankroll
        game_clock = game_state.game_clock
        round_num = game_state.round_num
        my_cards = round_state.hands[active]
        big_blind = bool(active)
        
    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. called NUM_ROUNDS times.
        Arguments:
        game_state:
        terminal_state:
        active:
        Returns:
        Nothing
        '''
        my_delta = terminal_state.deltas[active]
        previous_state = terminal_state.previous_state
        street = previous_state.street
        my_cards = previous_state.hands[active]
        opp_cards = previous_state.hands[1-active]
        
    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function
        Called any time the engine needs an action from your bot.
        Arguments:
        game_state:
        round_state:
        active:
        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        continue_cost = opp_pip - my_pip
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        net_upper_raise_bound = round_state.raise_bounds()
        stacks = [my_stack, opp_stack]
        
        my_action = None
        
        min_raise, max_raise = round_state.raise_bounds()
        pot_total = my_contribution + opp_contribution
        
        # raise logic
        if street < 3:
            raise_amount = int(my_pip + continue_cost + 0.4*(pot_total + continue_cost))
        else:
            raise_amount = int(my_pip + continue_cost + 0.75*(pot_total + continue_cost))
            
        # ensure raises are legal
        raise_amount = max([min_raise, raise_amount]) # getting the max of the min raise and the raise amount
        raise_amount = min([max_raise, raise_amount]) # getting the min of the max raise and the raise amount
        # we want to do this so that we don't raise more than the max raise or less than the min raise
        
        if (RaiseAction in legal_actions and (raise_amount <= my_stack)):
            temp_action = RaiseAction(raise_amount)
        elif (CallAction in legal_actions and (continue_cost <= my_stack)):
            temp_action = CallAction()
        elif CheckAction in legal_actions:
            temp_action = CheckAction()
        else:
            temp_action = FoldAction()
        
        _MONTE_CARLO_ITERS = 100
        
        # running monte carlo simulation when we have community cards vs when we don't
        if street < 3:
            strength = self.calc_strength(my_cards, _MONTE_CARLO_ITERS)
        else:
            strength = self.calc_strength(my_cards, _MONTE_CARLO_ITERS, board_cards)
            
        if continue_cost > 0:
            _SCARY = 0
            if continue_cost > 6:
                _SCARY = 0.1
            if continue_cost > 15:
                _SCARY = 0.2
            if continue_cost > 50:
                _SCARY = 0.35
            
            strength = max(0, strength - _SCARY)
            pot_odds = continue_cost / (pot_total + continue_cost)
            
            if strength >= pot_odds:
                if strength > 0.5 and random.random() < strength:
                    my_action = temp_action
                else:
                    my_action = CallAction()
            else:
                my_action = FoldAction()
        else:
            if random.random() < strength:
                my_action = temp_action
            else:
                my_action = CheckAction()
        return my_action
    
if __name__ == '__main__':
    run_bot(Player(), parse_args())
                    

            
        