'''
Encapsulates game and round state information for the player.
'''
from collections import namedtuple
from .actions import FoldAction, CallAction, CheckAction, RaiseAction

GameState = namedtuple('GameState', ['bankroll', 'game_clock', 'round_num'])
TerminalState = namedtuple('TerminalState', ['deltas', 'previous_state'])

NUM_ROUNDS = 1000
STARTING_STACK = 400
BIG_BLIND = 2
SMALL_BLIND = 1

class RoundState(namedtuple('_RoundState', [
    'button',
    'street',
    'pips',
    'stacks',
    'hands',
    'deck',
    'previous_state'
])):
    '''
    Encodes the game tree for one round of poker.
    '''
    def showdown(self):
        '''
        Compares the players' hands and computes payoffs.
        '''
        # Since we removed 'bids', store only the previous_state and deltas
        return TerminalState([0, 0], self)

    def legal_actions(self):
        '''
        Returns the set of legal moves for the active player.
        '''
        active = self.button % 2
        continue_cost = self.pips[1 - active] - self.pips[active]
        if continue_cost == 0:
            # we can only raise the stakes if both players can afford it
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction} if bets_forbidden else {CheckAction, RaiseAction}
        # continue_cost > 0
        raises_forbidden = (continue_cost >= self.stacks[active] or self.stacks[1 - active] == 0)
        return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}

    def raise_bounds(self):
        '''
        Returns a tuple of the minimum and maximum legal raises.
        '''
        active = self.button % 2
        continue_cost = self.pips[1 - active] - self.pips[active]
        max_contribution = min(self.stacks[active], self.stacks[1 - active] + continue_cost)
        min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
        return (self.pips[active] + min_contribution, self.pips[active] + max_contribution)

    def proceed_street(self):
        '''
        Resets the players' pips and advances to the next betting street.
        '''
        if self.street == 5:
            return self.showdown()
        if self.street == 0:
            # Example jump to street=3 if you want to replicate flop dealing
            return RoundState(
                button=1,
                street=3,
                pips=[0, 0],
                stacks=self.stacks,
                hands=self.hands,
                deck=self.deck,
                previous_state=self
            )
        # Otherwise just increment street
        return RoundState(
            button=1,
            street=self.street + 1,
            pips=[0, 0],
            stacks=self.stacks,
            hands=self.hands,
            deck=self.deck,
            previous_state=self
        )

    def proceed(self, action):
        '''
        Advances the game tree by one action performed by the active player.
        '''
        active = self.button % 2

        if isinstance(action, FoldAction):
            delta = self.stacks[0] - STARTING_STACK if active == 0 else STARTING_STACK - self.stacks[1]
            return TerminalState([delta, -delta], self)

        if isinstance(action, CallAction):
            if self.button == 0:  # sb calls bb pre-flop scenario
                return RoundState(
                    button=1,
                    street=0,
                    pips=[BIG_BLIND, BIG_BLIND],
                    stacks=[STARTING_STACK - BIG_BLIND, STARTING_STACK - BIG_BLIND],
                    hands=self.hands,
                    deck=self.deck,
                    previous_state=self
                )
            # both players acted
            new_pips = list(self.pips)
            new_stacks = list(self.stacks)
            contribution = new_pips[1 - active] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            state = RoundState(
                button=self.button + 1,
                street=self.street,
                pips=new_pips,
                stacks=new_stacks,
                hands=self.hands,
                deck=self.deck,
                previous_state=self
            )
            return state.proceed_street()

        if isinstance(action, CheckAction):
            # If both players checked, proceed to next street
            if (self.street == 0 and self.button > 0) or self.button > 1:
                return self.proceed_street()
            # otherwise let the other player act
            return RoundState(
                button=self.button + 1,
                street=self.street,
                pips=self.pips,
                stacks=self.stacks,
                hands=self.hands,
                deck=self.deck,
                previous_state=self
            )

        # Finally, handle RaiseAction
        new_pips = list(self.pips)
        new_stacks = list(self.stacks)
        contribution = action.amount - new_pips[active]
        new_stacks[active] -= contribution
        new_pips[active] += contribution
        return RoundState(
            button=self.button + 1,
            street=self.street,
            pips=new_pips,
            stacks=new_stacks,
            hands=self.hands,
            deck=self.deck,
            previous_state=self
        )
