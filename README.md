# Fox and Hounds

## How to run

### Install necessary tools

```bash
sudo apt update && sudo apt install python3 pipx
pipx ensurepath
pipx install uv
```

### Sync the environment and run the project

```bash
uv sync
uv run python src/main.py
```

## Minimax player

`MinimaxPlayer` (`src/players/minimax_player.py`) searches the game tree with minimax and alpha-beta pruning
(Russell & Norvig, *AIMA* 4th ed., ch. 5). It is configured in `src/main.py`:

```python
from players import Heuristic, MinimaxPlayer

fox = MinimaxPlayer(depth=4)  # all heuristics
hounds = MinimaxPlayer(depth=6, heuristics=[Heuristic.DISTANCE, Heuristic.LINE])
```

- `depth` – number of plies (single moves) looked ahead; higher is stronger but exponentially slower.
- `heuristics` – which evaluation terms are summed at the search horizon (default: all).

Scores are always from the fox's perspective (fox maximizes, hounds minimize). A won/lost game scores
±10 000 regardless of the heuristics, and quicker wins score higher. Weights are constants at the top of the file.

### Heuristics (`Heuristic` enum)

| Heuristic | Side | Meaning |
|-----------|------|---------|
| `ADVANCEMENT` | Fox | +5 points per row the fox has advanced. |
| `MOBILITY` | Fox | +3 per available fox move; −15 if it has at most one move (close to being trapped). |
| `DISTANCE` | Hounds | Sum of \|dx\|+\|dy\| between each hound and the fox; the smaller the better for hounds. |
| `LINE` | Hounds | +5 for each pair of column-neighbouring hounds whose rows differ by at most 1 (a staggered line the fox cannot break). |
| `STRAGGLER` | Hounds | −100 if the rearmost hound is at least 2 rows behind every other hound (a full empty row between them); a staircase is not penalised. |

The "line break" rule (fox's y above every hound's y means the fox wins) is not a heuristic: it is detected as a
terminal state by `Board.get_winner()`.

Choosing a subset changes the playing style, e.g. only `ADVANCEMENT` makes the fox run greedily forward,
`MOBILITY` alone makes it avoid getting boxed in, and `DISTANCE` alone makes the hounds chase the fox
without keeping formation. The full descriptions (in Polish) are in `docs/heuristics.md`.

### Current minimax vs minimax findings:

| Fox depth | Hounds Depth | Winner |
|---|---|---|
| 2 | 2 | Fox |
| 2 | 4, 6, 8 | Hounds |
| 4 | 2 | Fox |
| 4 | 4, 6, 8 | Hounds |
| 6 | 2, 4 | Fox |
| 6 | 6, 8 | Hounds |
