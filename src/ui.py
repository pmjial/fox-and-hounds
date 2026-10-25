from types import TracebackType
from typing import Self

from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from board import Board


class UI:
    def __init__(self):
        self._live: Live = Live()

    def __enter__(self) -> Self:
        _ = self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        return self._live.__exit__(exc_type, exc_val, exc_tb)

    def update(self, board: Board):
        header = "  " + "".join([f" {c} " for c in range(board.size)])
        lines: list[str] = [header]
        for r in range(board.edge, -1, -1):
            row = f"{r} "
            for c in range(board.size):
                bg = "#FF8000" if (r + c) % 2 == 1 else "#000000"
                if (c, r) == board.fox_pos:
                    row += f"[#FFFF00 on {bg}] F [/]"
                elif (c, r) in board.hounds_pos:
                    row += f"[#00FF80 on {bg}] H [/]"
                else:
                    row += f"[on {bg}]   [/]"
            lines.append(row)

        board_panel = Panel("\n".join(lines), expand=False, box=box.SIMPLE)

        info_text = (
            f"Turn: {board.turn_count}\n"
            + f"Player: {board.current_turn.name}\n"
            + f"Fox: {board.fox_pos}\n"
            + f"Hounds: {board.hounds_pos}\n"
            + (f"Winner: {winner.name}\n" if (winner := board.get_winner()) else "")
        )
        info_panel = Panel(info_text, title="Info", width=48, box=box.SIMPLE)

        layout = Table.grid(padding=2)
        layout.add_row(board_panel, info_panel)
        self._live.update(layout)
