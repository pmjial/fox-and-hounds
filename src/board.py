from enum import Enum


class PlayerKind(Enum):
    Fox = 0
    Hounds = 1


Pos = tuple[int, int]
Move = tuple[Pos, Pos]


class Board:
    def __init__(self, size: int):
        if size % 2 == 1 or size < 4:
            raise ValueError('Board size must be an even number greater than "2"')
        self.size: int = size

        self.fox_pos: Pos = (0, 0)
        self.hounds_pos: list[Pos] = [(i, size - 1) for i in range(1, size, 2)]
        self.current_turn: PlayerKind = PlayerKind.Fox
        self.turn_count: int = 0

    @property
    def edge(self) -> int:
        return self.size - 1

    def get_winner(self) -> PlayerKind | None:
        if self.fox_pos[1] == self.edge or all(
            self.fox_pos[1] > hound_pos[1] for hound_pos in self.hounds_pos
        ):
            return PlayerKind.Fox

        if not self.get_legal_moves():
            return PlayerKind(not self.current_turn.value)

        return None

    def _is_legal_pos(self, pos: Pos) -> bool:
        x, y = pos
        return self.edge >= y >= 0 and self.edge >= x >= 0

    def _is_free_pos(self, pos: Pos) -> bool:
        return not (pos == self.fox_pos or pos in self.hounds_pos)

    def get_legal_moves(self) -> list[Move]:
        moves: list[Move] = []

        match self.current_turn:
            case PlayerKind.Fox:
                x, y = self.fox_pos
                for mx, my in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
                    mpos = (x + mx, y + my)
                    if self._is_legal_pos(mpos) and self._is_free_pos(mpos):
                        moves.append(Move(((x, y), mpos)))

            case PlayerKind.Hounds:
                for x, y in self.hounds_pos:
                    for mx in [-1, 1]:
                        mpos = (x + mx, y - 1)
                        if self._is_legal_pos(mpos) and self._is_free_pos(mpos):
                            moves.append(Move(((x, y), mpos)))

        return moves

    def make_move(self, move: Move) -> bool:
        if self.get_winner() or move not in self.get_legal_moves():
            return False

        if self.current_turn == PlayerKind.Fox:
            self.fox_pos = move[1]
            self.current_turn = PlayerKind.Hounds
        else:
            i = self.hounds_pos.index(move[0])
            self.hounds_pos[i] = move[1]
            self.hounds_pos.sort()
            self.current_turn = PlayerKind.Fox

        self.turn_count += 1

        return True

    def clone(self) -> "Board":
        copy = Board(self.size)
        copy.fox_pos = self.fox_pos
        copy.hounds_pos = self.hounds_pos.copy()
        copy.current_turn = self.current_turn
        copy.turn_count = self.turn_count
        return copy

    def to_ascii(self) -> str:
        sb = " " + "".join([f" {c} " for c in range(self.size)])
        for r in range(self.edge, -1, -1):
            row: str = str(r)
            for c in range(self.size):
                if (c, r) == self.fox_pos:
                    row += " F "
                elif (c, r) in self.hounds_pos:
                    row += " H "
                else:
                    row += " . "
            sb += "\n" + row

        return sb
