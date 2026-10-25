from time import sleep

from board import Board, PlayerKind
from players import RandomPlayer
from ui import UI


def main():
    fox = RandomPlayer()
    hounds = RandomPlayer()
    board = Board(8)

    with UI() as ui:
        ui.update(board)
        while not board.get_winner():
            player = fox if board.current_turn == PlayerKind.Fox else hounds
            move = player.get_move(board.clone())
            if board.make_move(move):
                ui.update(board)
                sleep(0.3)


if __name__ == "__main__":
    main()
