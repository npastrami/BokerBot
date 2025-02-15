import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import *

from engine import FoldAction, CheckAction, CallAction, RaiseAction

# NOTE: Cards are embedded as a single integer here.
# To interface with the way the engine is written we need to translate things

class CardEmbedding(nn.Module): 

    def __init__(self, dim):

        super(CardEmbedding, self).__init__()
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(52, dim)

    def forward(self , input):

        B, num_cards = input.shape

        x = input.view(-1)

        valid = x.ge(0).float() # −1 means ’no card’ 
        x = x.clamp(min=0)

        embs = self.card(x) + self.rank(x // 4) + self.suit(x % 4)
        embs = embs * valid.unsqueeze(1) # zero out ’no card ’ embeddings

        #sum across the cards in the hole / board
        return embs.view(B,num_cards, -1).sum(1)


class DeepCFRModel(nn.Module):

    def __init__( self , ncardtypes , nbets , nactions , dim=256):

        self.nbets = nbets
        self.nactions = nactions

        super(DeepCFRModel, self ).__init__()

        self.card_embeddings = nn.ModuleList( [CardEmbedding(dim) for i in range(ncardtypes)])

        self.card1 = nn.Linear(dim * ncardtypes, dim)
        self.card2 = nn.Linear(dim, dim)
        self.card3 = nn.Linear(dim, dim)

        self.bet1 = nn.Linear(nbets*2, dim)
        self.bet2 = nn.Linear(dim, dim)

        self.comb1 = nn.Linear(2 * dim, dim)
        self.comb2 = nn.Linear(dim, dim)
        self.comb3 = nn.Linear(dim, dim)

        self.action_head = nn.Linear(dim, nactions)

    def forward(self, cards, bets):
        """
        cards: ((N x 2), (N x 3)[, (N x 1), (N x 1)]) # (hole, board, [turn, river]) bets: N x nbet feats
        """

        card_embs = []
        
        for embedding, card_group in zip (self.card_embeddings, cards):

            card_embs.append(embedding(card_group))

        card_embs = torch.cat(card_embs, dim=1)

        x = F.relu(self.card1(card_embs)) 
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))

        # 1. bet branch
        bet_size = bets.clamp(0, 1e6)
        bet_occurred = bets.ge(0)
        bet_feats = torch.cat([ bet_size , bet_occurred.float()] , dim=1)


        y = F.relu(self.bet1(bet_feats)) 
        y = F.relu(self.bet2(y) + y)

        # 3. combined trunk
        z = torch.cat([x, y], dim=1)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)

        z = F.normalize(z) # (z − mean) / std return self . action head (z)
        return self.action_head(z)
    
    def get_card_num(self, card):

        card_num = (card.rank-1) + 13*(card.suit)

        return torch.tensor([card_num])

    def tensorize_cards(self, my_hand, board):

        hole_tensor = torch.tensor([self.get_card_num(card) for card in my_hand]).unsqueeze(0)

        board_nums = [self.get_card_num(card) for card in board]

        while len(board_nums) < 5:
            board_nums.append(-1)

        flop_tensor = torch.tensor(board_nums[:3]).unsqueeze(0)
        turn_tensor = torch.tensor([board_nums[3]]).unsqueeze(0)
        river_tensor = torch.tensor([board_nums[4]]).unsqueeze(0)

        return [hole_tensor, flop_tensor, turn_tensor, river_tensor]

    def tensorize_bets(self, bets):

        last_bets = bets[:][-self.nbets:]

        while len(last_bets) < self.nbets:
            last_bets.append(-1)

        tensorized_bets = torch.tensor(last_bets, dtype=torch.float32).unsqueeze(0)

        return tensorized_bets

    def tensorize_roundstate(self, roundstate, active):

        bets = roundstate.bets
        bet_tensor = self.tensorize_bets(bets)

        my_cards = roundstate.hands[active]  # your cards
        board_cards = roundstate.deck[:roundstate.street]  # the board cards

        card_tensor = self.tensorize_cards(my_cards, board_cards)

        return (card_tensor, bet_tensor)
    
    def tensorize_mask(self, round_state):

        min_raise, max_raise = round_state.raise_bounds()
        legal_actions = round_state.legal_actions()
        mask_tensor = torch.zeros(self.nactions, dtype=torch.float32)

        pot = sum([STARTING_STACK - round_state.stacks[i] for i in [0,1]])

        if FoldAction in legal_actions:
            mask_tensor[0] = 1
        
        if CheckAction in legal_actions:
            mask_tensor[1] = 1
        
        if CallAction in legal_actions:
            mask_tensor[2] = 1
        
        if RaiseAction in legal_actions:

            if min_raise <= math.ceil(pot*1/2) <= max_raise:
                mask_tensor[3] = 1
            if min_raise <= math.ceil(pot*3/2) <= max_raise:
                mask_tensor[4] = 1
        
        return mask_tensor.unsqueeze(0)