'''
The infrastructure for interacting with the engine.
'''
import argparse
import socket
from .actions import FoldAction, CallAction, CheckAction, RaiseAction, BidAction
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
        elif isinstance(action, BidAction): 
            code = 'A' + str(action.amount)
        else:  # isinstance(action, RaiseAction)
            code = 'R' + str(action.amount)
        self.socketfile.write(code + '\n')
        self.socketfile.flush()

    def get_base_state(self, state):
        """Helper method to get the base RoundState"""
        if isinstance(state, TerminalState):
            return state.previous_state
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
                    game_state = GameState(game_state.bankroll, float(clause[1:]), game_state.round_num)
                elif clause[0] == 'P':
                    active = int(clause[1:])
                elif clause[0] == 'H':
                    hands = [[], []]
                    hands[active] = clause[1:].split(',')
                    pips = [SMALL_BLIND, BIG_BLIND]
                    stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
                    round_state = RoundState(0, 0, False, [None, None], pips, stacks, hands, [], None)
                    if round_flag:
                        self.pokerbot.handle_new_round(game_state, round_state, active)
                        round_flag = False
                elif clause[0] == 'F':
                    base_state = self.get_base_state(round_state)
                    round_state = base_state.proceed(FoldAction())
                elif clause[0] == 'C':
                    base_state = self.get_base_state(round_state)
                    round_state = base_state.proceed(CallAction())
                elif clause[0] == 'K':
                    base_state = self.get_base_state(round_state)
                    round_state = base_state.proceed(CheckAction())
                elif clause[0] == 'R':
                    base_state = self.get_base_state(round_state)
                    round_state = base_state.proceed(RaiseAction(int(clause[1:])))
                elif clause[0] == 'A': 
                    base_state = self.get_base_state(round_state)
                    round_state = base_state.proceed(BidAction(int(clause[1:])))
                elif clause[0] == 'N':
                    hands = [[], []]
                    stacks, bids, active_hands = clause[1:].split('_')
                    bids = bids.split(',')
                    stacks = stacks.split(',')
                    hands[active] = active_hands.split(',')
                    base_state = self.get_base_state(round_state)
                    round_state = RoundState(
                        base_state.button,
                        base_state.street,
                        base_state.auction,
                        [int(x) for x in bids],
                        base_state.pips,
                        [int(x) for x in stacks],
                        hands,
                        [],
                        base_state
                    )
                elif clause[0] == 'B':
                    base_state = self.get_base_state(round_state)
                    round_state = RoundState(
                        base_state.button,
                        base_state.street,
                        base_state.auction,
                        base_state.bids,
                        base_state.pips,
                        base_state.stacks,
                        base_state.hands,
                        clause[1:].split(','),
                        base_state
                    )
                elif clause[0] == 'O':
                    # backtrack
                    base_state = self.get_base_state(round_state)
                    base_state = base_state.previous_state
                    revised_hands = list(base_state.hands)
                    revised_hands[1-active] = clause[1:].split(',')
                    # rebuild history
                    round_state = RoundState(
                        base_state.button,
                        base_state.street,
                        base_state.auction,
                        base_state.bids,
                        base_state.pips,
                        base_state.stacks,
                        revised_hands,
                        base_state.deck,
                        base_state.previous_state
                    )
                    round_state = TerminalState([0, 0], round_state.bids, round_state)
                elif clause[0] == 'D':
                    assert isinstance(round_state, TerminalState)
                    delta = int(clause[1:])
                    deltas = [-delta, -delta]
                    deltas[active] = delta
                    round_state = TerminalState(deltas, round_state.bids, round_state.previous_state)
                    game_state = GameState(game_state.bankroll + delta, game_state.game_clock, game_state.round_num)
                    self.pokerbot.handle_round_over(game_state, round_state, active)
                    game_state = GameState(game_state.bankroll, game_state.game_clock, game_state.round_num + 1)
                    round_flag = True
                elif clause[0] == 'Q':
                    return
            if round_flag:  # ack the engine
                base_state = self.get_base_state(round_state)
                self.send(CheckAction())
            else:
                base_state = self.get_base_state(round_state)
                assert active == base_state.button % 2
                action = self.pokerbot.get_action(game_state, base_state, active)
                self.send(action)


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