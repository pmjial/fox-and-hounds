import os

from dotenv import load_dotenv

from board import Board, Move, PlayerKind
from players import LLMPlayer, RandomPlayer
from ui import UI


def main():
    _ = load_dotenv()
    fox = LLMPlayer(os.getenv("PBAD_LLM_NAME") or "ingloryon/k2-horizon:7b")
    hounds = RandomPlayer()
    board = Board(8)
    history: list[tuple[PlayerKind, Move, str | None]] = []

    with UI() as ui:
        ui.update(board)
        while not board.get_winner():
            turn = board.current_turn
            player = fox if turn == PlayerKind.Fox else hounds
            move = player.get_move(board.clone())
            history.append((turn, move, player.analysis))
            if board.make_move(move):
                comment = ""
                for last_turn, (move_from, move_to), analysis in history[-2:]:
                    comment += (
                        f"\n[bold]Last {last_turn.name} move[/]: {move_from} -> {move_to}\n"
                        + (f"{analysis}\n" if analysis else "")
                    )
                ui.update(board, comment=comment)


if __name__ == "__main__":
    main()
