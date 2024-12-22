"""
The infrastructure for interacting with the engine.
"""
import argparse
import socket
from .actions import FoldAction, CallAction, CheckAction, RaiseAction
from .states import GameState, TerminalState, RoundState
from .states import STARTING_STACK, BIG_BLIND, SMALL_BLIND
from .bot import Bot

class Runner():
    '''
    Interacts with the engine.
    '''

    def __init__(self, pokerbot, socketfile):
        self.pokerbot = pokerbot
        self.socketfile = socketfile

    def receive(self):
        '''
        Generator for incoming messages from the engine.
        '''
        while True:
            packet = self.socketfile.readline().strip().split(' ')
            if not packet:
                break
            yield packet

    def send(self, action):
        '''
        Encodes an action and sends it to the engine.
        '''
        if isinstance(action, FoldAction):
            code = 'F'
        elif isinstance(action, CallAction):
            code = 'C'
        elif isinstance(action, CheckAction):
            code = 'K'
        else:  # isinstance(action, RaiseAction)
            code = 'R' + str(action.amount)
        self.socketfile.write(code + '\n')
        self.socketfile.flush()

    def get_round_state(self, state):
        '''
        Safely retrieve the underlying RoundState from either a RoundState or a TerminalState.
        If it's TerminalState, return its .previous_state (which should be a RoundState).
        '''
        if isinstance(state, TerminalState):
            return state.previous_state  # step back to the last non-terminal state
        return state

    def run(self):
        '''
        Reconstructs the game tree based on the action history received from the engine.
        '''
        game_state = GameState(0, 0., 1)
        round_state = None
        active = 0
        round_flag = True

        for packet in self.receive():
            for clause in packet:
                if clause[0] == 'T':
                    # T<time> => update game clock
                    game_state = GameState(
                        bankroll=game_state.bankroll,
                        game_clock=float(clause[1:]),
                        round_num=game_state.round_num
                    )

                elif clause[0] == 'P':
                    # P<seat> => which player is active?
                    active = int(clause[1:])

                elif clause[0] == 'H':
                    # H<cards> => new round, your hole cards
                    hands = [[], []]
                    hands[active] = clause[1:].split(',')
                    pips = [SMALL_BLIND, BIG_BLIND]
                    stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
                    # Create a fresh RoundState with no 'auction' or 'bids'
                    round_state = RoundState(
                        button=0,       # or some logic if you want to alternate
                        street=0,
                        pips=pips,
                        stacks=stacks,
                        hands=hands,
                        deck=[],
                        previous_state=None
                    )
                    if round_flag:
                        self.pokerbot.handle_new_round(game_state, round_state, active)
                        round_flag = False

                elif clause[0] == 'F':
                    # F => fold
                    base_state = self.get_round_state(round_state)
                    if base_state is not None:
                        round_state = base_state.proceed(FoldAction())

                elif clause[0] == 'C':
                    # C => call
                    base_state = self.get_round_state(round_state)
                    if base_state is not None:
                        round_state = base_state.proceed(CallAction())

                elif clause[0] == 'K':
                    # K => check
                    base_state = self.get_round_state(round_state)
                    if base_state is not None:
                        round_state = base_state.proceed(CheckAction())

                elif clause[0] == 'R':
                    # R<amount> => raise
                    amount = int(clause[1:])
                    base_state = self.get_round_state(round_state)
                    if base_state is not None:
                        round_state = base_state.proceed(RaiseAction(amount))

                # Removed 'A' (BidAction) and 'N' (auction) clauses, since no bids/auction logic

                elif clause[0] == 'B':
                    # B<boardcards> => new board / deck
                    base_state = self.get_round_state(round_state)
                    if base_state is not None:
                        round_state = RoundState(
                            button=base_state.button,
                            street=base_state.street,
                            pips=base_state.pips,
                            stacks=base_state.stacks,
                            hands=base_state.hands,
                            deck=clause[1:].split(','),  # new board
                            previous_state=round_state  # store transition
                        )

                elif clause[0] == 'O':
                    # O<opponent_cards> => opponent revealed cards, backtrack, then rebuild
                    base_state = self.get_round_state(round_state)
                    if base_state is not None:
                        # step back one previous state
                        prev_base = self.get_round_state(base_state.previous_state)
                        if prev_base is not None:
                            revised_hands = list(prev_base.hands)
                            revised_hands[1 - active] = clause[1:].split(',')
                            # Rebuild
                            rebuilt = RoundState(
                                button=prev_base.button,
                                street=prev_base.street,
                                pips=prev_base.pips,
                                stacks=prev_base.stacks,
                                hands=revised_hands,
                                deck=prev_base.deck,
                                previous_state=prev_base.previous_state
                            )
                            # Transition to a TerminalState
                            round_state = TerminalState(deltas=[0, 0],
                                                        previous_state=rebuilt)

                elif clause[0] == 'D':
                    # D<delta> => round ended, final delta for active seat
                    assert isinstance(round_state, TerminalState)
                    delta = int(clause[1:])
                    deltas = [-delta, -delta]
                    deltas[active] = delta
                    # rebuild TerminalState with final deltas
                    round_state = TerminalState(deltas=deltas,
                                                previous_state=round_state.previous_state)
                    # update bankroll
                    game_state = GameState(
                        bankroll=game_state.bankroll + delta,
                        game_clock=game_state.game_clock,
                        round_num=game_state.round_num
                    )
                    # handle end-of-round
                    self.pokerbot.handle_round_over(game_state, round_state, active)
                    # increment round number
                    game_state = GameState(
                        bankroll=game_state.bankroll,
                        game_clock=game_state.game_clock,
                        round_num=game_state.round_num + 1
                    )
                    round_flag = True

                elif clause[0] == 'Q':
                    # Q => engine says quit
                    return

            # End of packet processing:
            if round_flag:
                # Acknowledge the engine for a new round
                self.send(CheckAction())
            else:
                # If we haven't started a new round, ask the bot for the next action
                if isinstance(round_state, TerminalState):
                    # Already terminal => send a dummy Check to avoid timeouts
                    self.send(CheckAction())
                    round_flag = True
                else:
                    base_state = self.get_round_state(round_state)
                    if base_state is not None:
                        # optional seat mismatch fix
                        if active != (base_state.button % 2):
                            # auto-fix mismatch by forcing button = active
                            base_state = RoundState(
                                button=active,
                                street=base_state.street,
                                pips=base_state.pips,
                                stacks=base_state.stacks,
                                hands=base_state.hands,
                                deck=base_state.deck,
                                previous_state=base_state.previous_state
                            )
                            round_state = base_state

                        action = self.pokerbot.get_action(game_state, base_state, active)
                        self.send(action)
                    else:
                        # If something is out-of-sync, just send a Check
                        self.send(CheckAction())
                        round_flag = True

def parse_args():
    '''
    Parses arguments corresponding to socket connection information.
    '''
    parser = argparse.ArgumentParser(prog='python3 player.py')
    parser.add_argument('--host', type=str, default='localhost', help='Host to connect to, defaults to localhost')
    parser.add_argument('port', type=int, help='Port on host to connect to')
    return parser.parse_args()

def run_bot(pokerbot, args):
    '''
    Runs the pokerbot.
    '''
    assert isinstance(pokerbot, Bot)
    try:
        sock = socket.create_connection((args.host, args.port))
    except OSError:
        print('Could not connect to {}:{}'.format(args.host, args.port))
        return
    socketfile = sock.makefile('rw')
    runner = Runner(pokerbot, socketfile)
    runner.run()
    socketfile.close()
    sock.close()
