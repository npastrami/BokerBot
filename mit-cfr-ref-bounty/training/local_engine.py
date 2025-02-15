'''
6.9630 MIT POKERBOTS GAME ENGINE
DO NOT REMOVE, RENAME, OR EDIT THIS FILE
'''
from collections import namedtuple
import random


import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import *

import eval7
from engine import RoundState, FoldAction, CheckAction, CallAction, RaiseAction, TerminalState
from engine import STREET_NAMES,  CCARDS, PCARDS, PVALUE, STATUS
from training.local_player import LocalPlayer
from training.local_roundstate import LocalRoundState

class LocalGame():
    '''
    Manages logging and the high-level game procedure.
    '''

    def __init__(self):
        self.log = ['6.9630 MIT Pokerbots - ' + PLAYER_1_NAME + ' vs ' + PLAYER_2_NAME]

    def get_bounties(self):

        cardNames = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        bounties = [cardNames[random.randint(0, 12)], cardNames[random.randint(0, 12)]]

        return bounties
    
    def query(self, players, round_state):
        '''
        Requests one action from the pokerbot.

        This method handles communication with the bot, sending the current game state
        and receiving the bot's chosen action.

        Args:
            round_state (RoundState or TerminalState): The current state of the game.
            players (List): A list of the players in the game
        Returns:
            Action: One of FoldAction, CallAction, CheckAction, or RaiseAction representing
            the bot's chosen action. If the bot fails to provide a valid action, returns:
                - CheckAction if it's a legal move
                - FoldAction if check is not legal

        Notes:
            - Invalid or illegal actions are not executed
            - At the end of a round, only CheckAction is considered legal
        '''


        legal_actions = round_state.legal_actions() if isinstance(round_state, LocalRoundState) else {CheckAction}
        active = round_state.button%2
        player = players[active]

        action = player.get_action(round_state, active) 

        if isinstance(action, tuple(legal_actions)):
            # print("LEGAL ACTION")
            if isinstance(action, RaiseAction):
                amount = action.amount
                min_raise, max_raise = round_state.raise_bounds()
                if min_raise <= amount <= max_raise:
                    return action
            else:
                return action
            
        return CheckAction() if CheckAction in legal_actions else FoldAction()

    def run_round(self, players, bounties):
        '''
        Runs one round of poker.
        '''

        for player in players:
            player.handle_new_round()

        bets = []

        deck = eval7.Deck()
        deck.shuffle()
        hands = [deck.deal(2), deck.deal(2)]
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        round_state = LocalRoundState(0, 0, pips, stacks, hands, deck, bounties, None, bets)

        while not isinstance(round_state, TerminalState):
            action = self.query(players, round_state)
            round_state = round_state.proceed(action)
      
        for player,delta in zip(players,round_state.deltas):
            player.bankroll += delta


    def generate_roundstates(self, players, bounties):
        '''
        Runs one round of poker.
        '''

        round_state_log = []

        for player in players:
            player.handle_new_round()

        deck = eval7.Deck()
        deck.shuffle()
        hands = [deck.deal(2), deck.deal(2)]
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]

        bounties = self.get_bounties()
        round_state = LocalRoundState(0, 0, pips, stacks, hands, deck, bounties, None, [])
        while not isinstance(round_state, TerminalState):
            active = round_state.button % 2

            player = players[active]
            action = self.query(players, round_state)
            round_state_log.append(round_state)
            round_state = round_state.proceed(action)
        
        return round_state_log
    
    def simulate_round_state(self, players, roundstate):    

        if isinstance(roundstate, TerminalState):
            return roundstate.deltas

        roundstate.deck.shuffle()

        while not isinstance(roundstate, TerminalState):
            action = self.query(players, roundstate)
            roundstate = roundstate.proceed(action)
        
        return roundstate.deltas
    
    
    def run(self, players):
        '''
        Runs one game of poker.
        '''
        print('   __  _____________  ___       __           __        __    ')
        print('  /  |/  /  _/_  __/ / _ \\___  / /_____ ____/ /  ___  / /____')
        print(' / /|_/ // /  / /   / ___/ _ \\/  \'_/ -_) __/ _ \\/ _ \\/ __(_-<')
        print('/_/  /_/___/ /_/   /_/   \\___/_/\\_\\\\__/_/ /_.__/\\___/\\__/___/')
        print()
        print('Starting the Pokerbots engine...')

        for round_num in range(1, NUM_ROUNDS + 1):
            self.log.append('')
            self.log.append('Round #' + str(round_num) + STATUS(players))
            if round_num % ROUNDS_PER_BOUNTY == 1:
                cardNames = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
                bounties = [cardNames[random.randint(0, 12)], cardNames[random.randint(0, 12)]]
                self.log.append(f"Bounties reset to {bounties[0]} for player {players[0].name} and {bounties[1]} for player {players[1].name}")
            self.run_round(players, bounties)
            self.log.append('Winning counts at the end of the round: ' + STATUS(players))

            players = players[::-1]
            bounties = bounties[::-1]
        self.log.append('')
        self.log.append('Final' + STATUS(players))
        # for player in players:
        #     player.stop()
        name = GAME_LOG_FILENAME + '.txt'
        print('Writing', name)
        with open(name, 'w') as log_file:
            log_file.write('\n'.join(self.log))


if __name__ == '__main__':

    player_0 = LocalPlayer("Player A")
    player_1 = LocalPlayer("Player B")

    players = [player_0, player_1]
    LocalGame(players).run()

        





