from random import choice
from typing import override
from time import sleep

from board import Board, Move

from .base_player import BasePlayer


class RandomPlayer(BasePlayer):
    @override
    def get_move(self, board: Board) -> Move:
        sleep(0.5) # Added for game readability
        return choice(board.get_legal_moves())
