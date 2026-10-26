from collections import defaultdict
from typing import Self, override

from openai import OpenAI
from pydantic import BaseModel

from board import Board, Move, PlayerKind, Pos
from config import Config, ModelConfig

from .base_player import BasePlayer


class Decision(BaseModel):
    analysis: str
    move_id: int


class LLMPlayer(BasePlayer):
    def __init__(self, model_cfg: ModelConfig, *, timeout: float = 120.0):
        self.model_cfg = model_cfg
        super().__init__()
        self.client: OpenAI = OpenAI(
            base_url=model_cfg.base_url,
            api_key=model_cfg.api_key,
        )
        self.timeout: float = timeout

    @classmethod
    def load(cls, name: str, file: str = "config.json") -> Self:
        config = Config.load(file)
        if not (model_cfg := config.get(name)):
            raise ValueError(f"No such key {name} in {file}")
        return cls(model_cfg)

    @override
    def get_move(self, board: Board) -> Move:
        system_prompt = f"""
You are an expert strategist specializing in the game of "Fox and Hounds"

--- BOARD COORDINATES ---

- The board is indexed from 0 to {board.edge}.
- (0, 0) is the BOTTOM-LEFT corner.
- Row index (r) increases upwards (0 is bottom row, {board.edge} is top row).
- Column index (c) increases to the right (0 is leftmost, {board.edge} is rightmost).
- Positions and moves use (x, y) = (column, row).

--- GAME RULES ---

1. Board contains 1 fox and {len(board.hounds_pos)} hounds. One hound moves per hounds turn.
2. NO CAPTURING OR JUMPING: Pieces never jump over each other and never capture. They only block fields.
3. MOVEMENT:
    - FOX (F): Can move 1 step diagonally in ANY of the 4 directions (up-left, up-right, bottom-left, bottom-right).
    - HOUND (H): Can move ONLY diagonally DOWN (y - 1). Hounds DON'T move backward.
4. WIN CONDITIONS:
    - FOX (F): wins if it reaches Row {board.edge} OR slips past all hounds (its y coordinate is greater than every hound's).
    - HOUND (H): win if they completely surround the fox so it has no legal moves left on its turn.

--- YOUR TASK ---

Analyze current board state carefully. You will play as either the "Fox" or "Hounds".
You must perform your turn and decide on the move that will give you the best advantage.
In analysis, briefly explain your chosen move in one or two short sentences.
"""

        legal_moves_text = ""
        legal_moves = board.get_legal_moves()
        id = 1
        match board.current_turn:
            case PlayerKind.Fox:
                for _, move_to in legal_moves:
                    legal_moves_text += f"{id}. {move_to}\n"
                    id += 1
            case PlayerKind.Hounds:
                hound_moves_map: dict[Pos, list[Pos]] = defaultdict(list)
                for move_from, move_to in legal_moves:
                    hound_moves_map[move_from].append(move_to)

                for move_from, moves_to in hound_moves_map.items():
                    legal_moves_text += f"Hound at {move_from} can move to:\n"
                    for move_to in moves_to:
                        legal_moves_text += f"  {id}. {move_to}\n"
                        id += 1

        user_prompt = f"""You MUST choose ONE legal move from the list below.
Respond ONLY with a JSON object matching this exact schema:
- move_id: integer ID of your chosen legal move
- analysis: short sentence explaining your move's strategic purpose, suitable for display

Choose only an ID listed below. Do not return move coordinates, additional fields, Markdown, or text outside the JSON object.
Make the explanation specific to the chosen move and current board; avoid claiming a guaranteed win unless it follows directly.

--- CURRENT BOARD STATE ---
TURN: {board.turn_count}
YOU PLAY AS: {board.current_turn.name}
FOX POSITION: {board.fox_pos}
HOUNDS POSITIONS: {board.hounds_pos}

LEGAL MOVES:
{legal_moves_text}

BOARD:
{board.to_ascii()}
        """

        response = self.client.chat.completions.create(
            model=self.model_cfg.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.model_cfg.temperature,
            response_format={"type": "json_object"},
            timeout=self.model_cfg.timeout,
        )
        if not (raw_content := response.choices[0].message.content):
            raise ValueError(f"{self.model} returned an invalid message shape")

        decision = Decision.model_validate_json(raw_content)
        if not (0 <= decision.move_id <= len(legal_moves)):
            raise ValueError(f"{self.model} returned an out-of-bounds move_id")

        self.analysis = decision.analysis
        return legal_moves[decision.move_id - 1]
