"""Minimax player with alpha-beta pruning

How a move is chosen:
1. `MinimaxPlayer.get_move` tries every legal move from the current position
   and asks `_search` how good the resulting position is.
2. `_search` recursively plays out the game `depth` plies ahead (one ply = one
   move by one side), alternating between the fox and the hounds.
3. When the depth runs out, or someone has won, the position is scored:
   a win/loss by a fixed huge number, otherwise by `evaluate`, which sums the
   enabled heuristics (see `docs/heuristics.md` and the README).
4. Alpha-beta pruning skips branches that provably cannot change the result.

All scores are from the FOX's point of view: higher is better for the fox,
lower is better for the hounds. So the fox is the MAX player and the hounds
are the MIN player, regardless of which side this player controls.
"""

from collections.abc import Collection
from enum import Enum
from math import inf
from time import sleep
from typing import override

from board import Board, Move, PlayerKind

from .base_player import BasePlayer

# Tunable weights
WIN_SCORE = 10_000  # win/loss score, must be greater than any sum of heuristics
# Fox
ADVANCEMENT_WEIGHT = 5  # Points per row the fox has advanced (counting from row 0 where he starts by default)
MOBILITY_WEIGHT = 3  # Points per available move (more room to maneuver is preferred)
LAST_MOVE_PENALTY = 15  # Subtracted when fox has only one move left (close to being trapped)
# Hounds
DISTANCE_WEIGHT = 1  # Weight of summed Manhattan distance between hounds and fox (the closer the better)
LINE_BONUS = 5  # Bonus for each pair of hounds neighboring themselves (keeping close formation is preferred)
STRAGGLER_PENALTY = (
    100  # Massive penalty for when ahound is too far from the others (MIN_STRAGGLER_GAP rows or more behind the rest)
)
MIN_STRAGGLER_GAP = 2  # Row distance between the rearmost hound and the next one that counts as a straggler


class Heuristic(Enum):
    """Evaluation terms that can be switched on/off per player."""

    # Fox-oriented
    ADVANCEMENT = "advancement"
    MOBILITY = "mobility"
    # Hounds-oriented
    DISTANCE = "distance"
    LINE = "line"
    STRAGGLER = "straggler"


def _fox_mobility(board: Board) -> int:
    """Number of moves the fox could make, even if it is not the fox's turn."""
    # `get_legal_moves` depends on whose turn it is, so probe a copy in which it is the fox's turn
    probe = board.clone()
    probe.current_turn = PlayerKind.Fox
    return len(probe.get_legal_moves())


def _advancement(board: Board) -> float:
    """Reward for every row the fox has moved forward (its y coordinate)."""
    return ADVANCEMENT_WEIGHT * board.fox_pos[1]


def _mobility(board: Board) -> float:
    """Small bonus per available fox move; penalty if it is nearly trapped."""
    moves = _fox_mobility(board)
    score = MOBILITY_WEIGHT * moves
    if moves <= 1:
        # With a single (or no) escape the fox is close to losing.
        score -= LAST_MOVE_PENALTY
    return score


def _distance(board: Board) -> float:
    """
    Sum of |dx| + |dy| between each hound and the fox.

    Hounds want this small, so the fox benefits from a large value.
    """
    fx, fy = board.fox_pos
    total = sum(abs(x - fx) + abs(y - fy) for x, y in board.hounds_pos)
    return DISTANCE_WEIGHT * total


def _line(board: Board) -> float:
    """
    Bonus for hounds that stay in a tight (staggered) line.

    Hounds are ordered by column; two neighbouring hounds are "steady" when
    their rows differ by at most 1. Such a staircase is hard for the fox to
    slip through.
    """
    hounds = sorted(board.hounds_pos)
    steady = sum(1 for a, b in zip(hounds, hounds[1:]) if abs(a[1] - b[1]) <= 1)
    return LINE_BONUS * steady


def _straggler(board: Board) -> float:
    """Huge penalty when one hound is left behind the others.

    Hounds only move down, so the rearmost hound (highest row) is a straggler
    when it is at least MIN_STRAGGLER_GAP rows above every other hound, i.e. at
    least one whole row separates it from the rest. A staircase is fine.
    """
    rows = sorted((y for _, y in board.hounds_pos), reverse=True)
    return STRAGGLER_PENALTY if rows[0] - rows[1] >= MIN_STRAGGLER_GAP else 0


# Maps each heuristic to (function, sign). The sign converts the magnitude into
# the fox's perspective: +1 = good for the fox, -1 = good for the hounds.
TERMS = {
    Heuristic.ADVANCEMENT: (_advancement, 1),  # fox gains from larger values
    Heuristic.MOBILITY: (_mobility, 1),  # fox gains from larger values
    Heuristic.DISTANCE: (_distance, 1),  # hounds want it small, so large is good for fox
    Heuristic.LINE: (_line, -1),  # a steady hound line is good for the hounds
    Heuristic.STRAGGLER: (_straggler, 1),  # straggler: it is a penalty for the hounds, so fox gains
}


class MinimaxPlayer(BasePlayer):
    """
    Depth-limited minimax with alpha-beta pruning.

    Args:
        depth: Search depth in plies (single moves, not full fox+hounds rounds). Larger is stronger but exponentially slower.
        heuristics: Which `Heuristic` terms to use; `None` enables all of them.
    """

    def __init__(
        self,
        depth: int = 4,
        heuristics: Collection[Heuristic] | None = None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("Depth must be at least 1")
        self.depth: int = depth
        self.heuristics: frozenset[Heuristic] = frozenset(heuristics or Heuristic)
        # Positions examined during the last `get_move` (reported in `analysis`).
        self.nodes: int = 0

    def evaluate(self, board: Board) -> float:
        """Heuristic score of a non-terminal position (fox's perspective)."""
        return sum(sign * term(board) for h, (term, sign) in TERMS.items() if h in self.heuristics)

    def _search(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        """
        Value of `board` searching `depth` more plies (MAX-VALUE / MIN-VALUE).

        alpha: best value the MAX player (fox) is already guaranteed elsewhere.
        beta:  best value the MIN player (hounds) is already guaranteed elsewhere.
        Once alpha >= beta the current branch cannot influence the decision made
        higher in the tree, so the remaining moves are skipped (pruned).
        """
        self.nodes += 1

        # Terminal position: the game is over, no need to look deeper
        if winner := board.get_winner():
            bonus = WIN_SCORE + depth  # Adding depth to prefer quicker wins
            return bonus if winner == PlayerKind.Fox else -bonus

        # Search horizon reached: fall back to the heuristic estimate
        if depth == 0:
            return self.evaluate(board)

        maximizing = board.current_turn == PlayerKind.Fox
        best = -inf if maximizing else inf
        for move in board.get_legal_moves():
            child = board.clone()
            child.make_move(move)
            value = self._search(child, depth - 1, alpha, beta)

            if maximizing:
                best = max(best, value)
                alpha = max(alpha, best)
            else:
                best = min(best, value)
                beta = min(beta, best)

            # The opponent above would never let the game reach this node
            if alpha >= beta:
                break
        return best

    @override
    def get_move(self, board: Board) -> Move:
        """Pick the best legal move for the side to move on `board`."""
        self.nodes = 0
        # The fox picks the move with the highest value, the hounds the lowest
        maximizing = board.current_turn == PlayerKind.Fox
        alpha, beta = -inf, inf
        best_value = -inf if maximizing else inf
        best_move: Move | None = None

        for move in board.get_legal_moves():
            child = board.clone()
            child.make_move(move)
            value = self._search(child, self.depth - 1, alpha, beta)

            # Strict comparison: on ties the first generated move is kept
            if (value > best_value) if maximizing else (value < best_value):
                best_value, best_move = value, move

            # Tighten the window so later root moves can be pruned too
            if maximizing:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)

        if best_move is None:
            raise ValueError("No legal moves available")

        # Shown by the UI next to the move
        self.analysis = f"Minimax (alpha-beta), depth {self.depth}: value {best_value:g}, {self.nodes} nodes visited."
        sleep(0.5)
        return best_move
