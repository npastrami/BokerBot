"""
Encapsulates game and round state information for the player.
"""
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
    'pips',           # [pips_p0, pips_p1]
    'stacks',         # [stack_p0, stack_p1]
    'hands',          # [hand_p0, hand_p1]
    'deck',           # board cards
    'previous_state'
])):
    """
    Encodes the game tree for one round of poker, without auction/bid functionality.
    """

    def showdown(self):
        """
        Concludes the hand, returning a TerminalState with no chip exchange by default.
        (Typically you'd compute who won, but here's a stub that returns deltas of [0, 0]).
        """
        return TerminalState([0, 0], self)

    def legal_actions(self):
        """
        Returns a set of the active player's legal moves (Fold/Call/Check/Raise)
        based on the needed continue_cost.
        """
        active = self.button % 2
        continue_cost = self.pips[1 - active] - self.pips[active]

        # If there's no cost to continue, the player can Check or Raise (unless stacks are empty).
        if continue_cost == 0:
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction} if bets_forbidden else {CheckAction, RaiseAction}
        else:
            # If there's a cost to continue, the player can Fold, Call, or possibly Raise
            raises_forbidden = (
                continue_cost >= self.stacks[active] or
                self.stacks[1 - active] == 0
            )
            return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}

    def raise_bounds(self):
        """
        Returns (min_raise, max_raise) in terms of total contribution.
        e.g. if min_raise = 10, that means the player's total pips after the raise is 10.
        """
        active = self.button % 2
        continue_cost = self.pips[1 - active] - self.pips[active]
        max_contribution = min(self.stacks[active], self.stacks[1 - active] + continue_cost)
        min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
        return (
            self.pips[active] + min_contribution,
            self.pips[active] + max_contribution
        )

    def proceed_street(self):
        """
        Resets the pips and advances to the next street (0 -> 3 -> 4 -> 5).
        If street == 5, it's a showdown -> TerminalState.
        """
        # If we reach street == 5, let's do showdown.
        if self.street == 5:
            return self.showdown()

        # Move from street=0 (preflop) to street=3 (flop), then to 4 (turn), 5 (river).
        next_street = 3 if self.street == 0 else self.street + 1

        # Reset pips to [0, 0], pass the rest along
        return RoundState(
            button=1,                 # normally you'd pass who is next to act; using 1 if you want to alternate
            street=next_street,
            pips=[0, 0],
            stacks=self.stacks,
            hands=self.hands,
            deck=self.deck,
            previous_state=self
        )

    def proceed(self, action):
        """
        Advances the game tree by one action performed by the active player.
        """
        active = self.button % 2
        continue_cost = self.pips[1 - active] - self.pips[active]

        # ---------------------------
        # 1) FOLD
        # ---------------------------
        if isinstance(action, FoldAction):
            # If active=0 folds, they lose their contributed chips (delta = stacks[0] - STARTING_STACK).
            # If active=1 folds, they lose their contributed chips (delta = stacks[1] - STARTING_STACK).
            delta = (
                self.stacks[0] - STARTING_STACK
                if active == 0 else
                STARTING_STACK - self.stacks[1]
            )
            return TerminalState([delta, -delta], self)

        # ---------------------------
        # 2) CALL
        # ---------------------------
        if isinstance(action, CallAction):
            # If button == 0 (SB calls BB) as a special preflop scenario
            if self.button == 0:
                # Example: both pips become BIG_BLIND, both stacks are (STARTING_STACK - BIG_BLIND)
                return RoundState(
                    button=1,        # now seat 1 acts next
                    street=0,        # still preflop
                    pips=[BIG_BLIND, BIG_BLIND],
                    stacks=[
                        STARTING_STACK - BIG_BLIND,
                        STARTING_STACK - BIG_BLIND
                    ],
                    hands=self.hands,
                    deck=self.deck,
                    previous_state=self
                )
            else:
                # Both players have matched contributions
                new_pips = list(self.pips)
                new_stacks = list(self.stacks)
                contribution = continue_cost
                new_stacks[active] -= contribution
                new_pips[active] += contribution

                # Move the button forward by 1
                state = RoundState(
                    button=self.button + 1,
                    street=self.street,
                    pips=new_pips,
                    stacks=new_stacks,
                    hands=self.hands,
                    deck=self.deck,
                    previous_state=self
                )
                # Once both have acted, proceed to the next street
                return state.proceed_street()

        # ---------------------------
        # 3) CHECK
        # ---------------------------
        if isinstance(action, CheckAction):
            # If preflop button > 0, or after the first check, both players have effectively checked
            if (self.street == 0 and self.button > 0) or (self.button > 1):
                return self.proceed_street()
            else:
                # Let the other player act
                return RoundState(
                    button=self.button + 1,
                    street=self.street,
                    pips=self.pips,
                    stacks=self.stacks,
                    hands=self.hands,
                    deck=self.deck,
                    previous_state=self
                )

        # ---------------------------
        # 4) RAISE
        # ---------------------------
        # (since we removed 'BidAction', this is the only bet-like option)
        if isinstance(action, RaiseAction):
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

        # If some unrecognized action occurs, just return self or raise an error
        # but typically we won't get here
        return self

