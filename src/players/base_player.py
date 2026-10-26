from abc import ABC, abstractmethod

from board import Board, Move


class BasePlayer(ABC):
    def __init__(self):
        self.analysis: str | None = None

    @abstractmethod
    def get_move(self, board: Board) -> Move: ...
