'''
6.9630 MIT POKERBOTS GAME ENGINE
DO NOT REMOVE, RENAME, OR EDIT THIS FILE
'''
from collections import namedtuple
from threading import Thread
from queue import Queue
import math

import eval7
import sys
import os
import random

sys.path.append(os.getcwd())
from config import * 

from engine import FoldAction, CallAction, CheckAction, RaiseAction, TerminalState
# FoldAction = namedtuple('FoldAction', [])
# CallAction = namedtuple('CallAction', [])
# CheckAction = namedtuple('CheckAction', [])
# # we coalesce BetAction and RaiseAction for convenience
# RaiseAction = namedtuple('RaiseAction', ['amount'])
# TerminalState = namedtuple('TerminalState', ['deltas', 'bounty_hits', 'previous_state'])

STREET_NAMES = ['Flop', 'Turn', 'River']
DECODE = {'F': FoldAction, 'C': CallAction, 'K': CheckAction, 'R': RaiseAction}
CCARDS = lambda cards: ','.join(map(str, cards))
PCARDS = lambda cards: '[{}]'.format(' '.join(map(str, cards)))
PVALUE = lambda name, value: ', {} ({})'.format(name, value)
STATUS = lambda players: ''.join([PVALUE(p.name, p.bankroll) for p in players])

# Socket encoding scheme:
#
# T#.### the player's game clock
# P# the- player's index
# H**,** the player's hand in common format
# F a fold action in the round history
# C a call action in the round history
# K a check action in the round history
# R### a raise action in the round history
# B**,**,**,**,** the board cards in common format
# O**,** the opponent's hand in common format
# D### the player's bankroll delta from the round
# Y## (both numbers 0 or 1 (or # which means masked): first is player hit bounty, second is opponent hit bounty)
#       Note: only winning player bounty hit is revealed (or both if split pot)
# Q game over
#
# Clauses are separated by spaces
# Messages end with '\n'
# The engine expects a response of K at the end of the round as an ack,
# otherwise a response which encodes the player's action
# Action history is sent once, including the player's actions


class LocalRoundState(namedtuple('_RoundState', ['button', 'street', 'pips', 'stacks', 'hands', 'deck', 'bounties', 'previous_state', 'bets'])):
    '''
    Encodes the game tree for one round of poker.
    '''
    def get_bounty_hits(self):
        '''
        Determines if each player hit their bounty card during the round.

        A bounty is hit if the player's bounty card rank appears in either:
        - Their hole cards
        - The community cards dealt so far

        Returns:
            tuple[bool, bool]: A tuple containing two booleans where:
                - First boolean indicates if Player 1's bounty was hit
                - Second boolean indicates if Player 2's bounty was hit
        '''
        cards0 = self.hands[0] + ([] if self.street == 0 else self.deck.peek(self.street))
        cards1 = self.hands[1] + ([] if self.street == 0 else self.deck.peek(self.street))
        cardNames = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        return (self.bounties[0] in [cardNames[card.rank] for card in cards0],
                self.bounties[1] in [cardNames[card.rank] for card in cards1])

    def get_delta(self, winner_index: int) -> int:
        '''Returns the delta after bounty rules are applied.

        Args:
            winner_index (int): Index of the winning player. Must be 0 (player A),
                1 (player B), or 2 (split pot).

        Returns:
            int: The delta value after applying bounty rules.
        '''
        assert winner_index in [0, 1, 2]

        bounty_hit_0, bounty_hit_1 = self.get_bounty_hits()

        delta = 0
        if winner_index == 2:
            # Case of split pots
            assert(self.stacks[0] == self.stacks[1]) # split pots only happen on the river + equal stacks
            delta = STARTING_STACK - self.stacks[0]
            if bounty_hit_0 and not bounty_hit_1:
                delta = delta * (BOUNTY_RATIO - 1) / 2 + BOUNTY_CONSTANT
            elif not bounty_hit_0 and bounty_hit_1:
                delta = -(delta * (BOUNTY_RATIO - 1) / 2 + BOUNTY_CONSTANT)
            else:
                delta = 0
        else:
            # Case of one player winning
            if winner_index == 0:
                delta = STARTING_STACK - self.stacks[1]
                if bounty_hit_0:
                    delta = delta * BOUNTY_RATIO + BOUNTY_CONSTANT
            else:
                delta = self.stacks[0] - STARTING_STACK
                if bounty_hit_1:
                    delta = delta * BOUNTY_RATIO - BOUNTY_CONSTANT

        # if delta is not an integer, round it down or up depending on who's in position
        if abs(delta - math.floor(delta)) > 1e-6:
            delta = math.floor(delta) if self.button % 2 == 0 else math.ceil(delta)
        return int(delta)


    def showdown(self) -> TerminalState:
        '''
        Compares the players' hands and computes the final payoffs at showdown.

        Evaluates both players' hands (hole cards + community cards) and determines
        the winner. The payoff (delta) is calculated based on:
        - The winner of the hand
        - Whether any bounties were hit
        - The current pot size

        Returns:
            TerminalState: A terminal state object containing:
                - List of deltas (positive for winner, negative for loser)
                - Tuple of bounty hit results for both players
                - Reference to the previous game state
        
        Note:
            This method assumes both players have equal stacks when reaching showdown,
            which is enforced by an assertion.
        '''
        score0 = eval7.evaluate(self.deck.peek(5) + self.hands[0])
        score1 = eval7.evaluate(self.deck.peek(5) + self.hands[1])
        assert(self.stacks[0] == self.stacks[1])
        if score0 > score1:
            delta = self.get_delta(0)
        elif score0 < score1:
            delta = self.get_delta(1)
        else:
            # split the pot
            delta = self.get_delta(2)
        
        return TerminalState([int(delta), -int(delta)], self.get_bounty_hits(), self)

    def legal_actions(self):
        '''
        Returns a set which corresponds to the active player's legal moves.
        '''
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        if continue_cost == 0:
            # we can only raise the stakes if both players can afford it
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction} if bets_forbidden else {CheckAction, RaiseAction}
        # continue_cost > 0
        # similarly, re-raising is only allowed if both players can afford it
        raises_forbidden = (continue_cost == self.stacks[active] or self.stacks[1-active] == 0)
        return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}

    def raise_bounds(self):
        '''
        Returns a tuple of the minimum and maximum legal raises.
        '''
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        max_contribution = min(self.stacks[active], self.stacks[1-active] + continue_cost)
        min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
        return (self.pips[active] + min_contribution, self.pips[active] + max_contribution)

    def proceed_street(self):
        '''
        Resets the players' pips and advances the game tree to the next round of betting.
        '''
        if self.street == 5:
            return self.showdown()
        new_street = 3 if self.street == 0 else self.street + 1

        new_bets = self.bets[:]
        new_bets.append(0)
        # print("The bets in proceed street are: ", new_bets)

        return LocalRoundState(1, new_street, [0, 0], self.stacks, self.hands, self.deck, self.bounties, self, new_bets)

    def proceed(self, action):
        '''
        Advances the game tree by one action performed by the active player.

        Args:
            action: The action being performed. Must be one of:
                - FoldAction: Player forfeits the hand
                - CallAction: Player matches the current bet
                - CheckAction: Player passes when no bet to match
                - RaiseAction: Player increases the current bet

        Returns:
            Either:
            - RoundState: The new state after the action is performed
            - TerminalState: If the action ends the hand (e.g., fold or final call)

        Note:
            The button value is incremented after each action to track whose turn it is.
            For FoldAction, the inactive player is awarded the pot.
            For CallAction on button 0, both players post blinds.
            For CheckAction, advances to next street if both players have acted.
            For RaiseAction, updates pips and stacks based on raise amount.
        '''

      
        new_bets = self.bets[:]
        # print("The new bets in proceed is: ", new_bets)
        active = self.button % 2

        if isinstance(action, FoldAction):
            # new_bets.append(0)
            delta = self.get_delta((1 - active) % 2) # if active folds, the other player (1 - active) wins
            return TerminalState([delta, -delta], self.get_bounty_hits(), self)

        if isinstance(action, CallAction):

            new_bets.append(0)
            if self.button == 0:  # sb calls bb
                return LocalRoundState(1, 0, [BIG_BLIND] * 2, [STARTING_STACK - BIG_BLIND] * 2, self.hands, self.deck, self.bounties, self, new_bets)
            # both players acted
            new_pips = list(self.pips)
            new_stacks = list(self.stacks)
            contribution = new_pips[1-active] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution

            state = LocalRoundState(self.button + 1, self.street, new_pips, new_stacks, self.hands, self.deck, self.bounties, self, new_bets)
            return state.proceed_street()
        if isinstance(action, CheckAction):
            if (self.street == 0 and self.button > 0) or self.button > 1:  # both players acted
                return self.proceed_street()
            new_bets.append(0)
            # let opponent act
            return LocalRoundState(self.button + 1, self.street, self.pips, self.stacks, self.hands, self.deck, self.bounties, self, new_bets)
        
        # Here its isinstance(action, RaiseAction):
        new_pips = list(self.pips)
        new_stacks = list(self.stacks)
        contribution = action.amount - new_pips[active]
        new_stacks[active] -= contribution
        new_pips[active] += contribution
        new_bets.append(action.amount)
        return LocalRoundState(self.button + 1, self.street, new_pips, new_stacks, self.hands, self.deck, self.bounties, self, new_bets)
