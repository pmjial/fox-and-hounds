from board import Board, Move, PlayerKind
from players import LLMPlayer, MinimaxPlayer, RandomPlayer
from ui import UI


def main():
    fox = LLMPlayer.load(
        "Groq-alam",
        # "Qwen3.6-35B-A3B-MTP",
        "allam-2-7b",
        # "K2-Horizon-3.7B",
    )

    hounds = MinimaxPlayer(depth=2)
    # MinimaxPlayer(depth=4, heuristics=[Heuristic.ADVANCEMENT, ...]) selects heuristics

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
                    comment += f"\n[bold]Last {last_turn.name} move[/]: {move_from} -> {move_to}\n" + (
                        f"{analysis}\n" if analysis else ""
                    )
                ui.update(board, comment=comment)


if __name__ == "__main__":
    main()
