from abc import ABC, abstractmethod

from board import Board, Move


class BasePlayer(ABC):
    @abstractmethod
    def get_move(self, board: Board) -> Move: ...
