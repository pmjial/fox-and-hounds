from random import choice
from typing import override

from board import Board, Move

from .base_player import BasePlayer


class RandomPlayer(BasePlayer):
    @override
    def get_move(self, board: Board) -> Move:
        return choice(board.get_legal_moves())
