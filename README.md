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

## LLM player variants

`LLMPlayer` defaults to direct reasoning with stateless context. The two options are independent, giving four combinations:

| Option | Value | Behavior |
|---|---|---|
| `ReasoningMode` | `DIRECT` | Make one request to choose a legal move and provide the UI analysis. |
| `ReasoningMode` | `ANALYSIS` | First request a separate threat analysis, then make a second request that receives the analysis and chooses the move. This adds an API call per turn. |
| `ContextMode` | `STATELESS` | Each turn starts with the system prompt and current position only; no earlier turn messages are retained. |
| `ContextMode` | `CONVERSATION` | Keep that player's prompt/response history across turns. Each turn still includes the full current board and legal moves, plus the last opponent move when available. |

Set either option when constructing or loading the player:

```python
from players import ContextMode, LLMPlayer, ReasoningMode

# Analysis + conversation; other combinations use the same options independently.
fox = LLMPlayer.load(
    "Local",
    "Gemma4-12B-QAT-MTP",
    reasoning=ReasoningMode.ANALYSIS,
    context=ContextMode.CONVERSATION,
)
```

Conversation history belongs to one `LLMPlayer` instance. Create a fresh instance for each game to avoid carrying history between matches.

If the provider reports `json_validate_failed`, `LLMPlayer` retries once with a request for shorter, complete JSON. This applies separately to threat analysis and move selection. Move selection shares its single retry between JSON-generation failures and out-of-range move IDs. If the retry fails, the error is surfaced rather than choosing an arbitrary move. Other API errors are not retried by this logic. JSON support and successful generation depend on the provider and model.

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
