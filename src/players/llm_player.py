from collections import defaultdict
from enum import Enum
from typing import Self, override

from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from board import Board, Move, PlayerKind, Pos
from config import Config, ModelConfig

from .base_player import BasePlayer


class ReasoningMode(Enum):
    DIRECT = "direct"
    ANALYSIS = "analysis"


class ContextMode(Enum):
    STATELESS = "stateless"
    CONVERSATION = "conversation"


class Decision(BaseModel):
    analysis: str
    move_id: int


class ThreatAnalysis(BaseModel):
    analysis: str


class LLMPlayer(BasePlayer):
    def __init__(
        self,
        model_cfg: ModelConfig,
        *,
        timeout: float = 120.0,
        reasoning: ReasoningMode = ReasoningMode.DIRECT,
        context: ContextMode = ContextMode.STATELESS,
    ):
        self.model_cfg = model_cfg
        self.reasoning = reasoning
        self.context = context
        super().__init__()
        self.client: OpenAI = OpenAI(
            base_url=model_cfg.conn.base_url,
            api_key=model_cfg.conn.api_key.get_secret_value(),
        )
        self.timeout: float = timeout
        self._conversation: list[ChatCompletionMessageParam] = []
        self._board_after_own_move: Board | None = None

    @classmethod
    def load(
        cls,
        conn_name: str,
        model_name: str,
        file: str = "config.json",
        *,
        reasoning: ReasoningMode = ReasoningMode.DIRECT,
        context: ContextMode = ContextMode.STATELESS,
    ) -> Self:
        config = Config.load(file)
        if not (conn := config.get(conn_name)):
            raise ValueError(f"No such key {conn_name} in {file}")
        return cls(ModelConfig(name=model_name, conn=conn), reasoning=reasoning, context=context)

    def _system_prompt(self, board: Board) -> str:
        return f"""
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
"""

    @staticmethod
    def _legal_moves_text(board: Board, legal_moves: list[Move]) -> str:
        legal_moves_text = ""
        move_id = 1
        match board.current_turn:
            case PlayerKind.Fox:
                for _, move_to in legal_moves:
                    legal_moves_text += f"{move_id}. {move_to}\n"
                    move_id += 1
            case PlayerKind.Hounds:
                hound_moves_map: dict[Pos, list[Pos]] = defaultdict(list)
                for move_from, move_to in legal_moves:
                    hound_moves_map[move_from].append(move_to)

                for move_from, moves_to in hound_moves_map.items():
                    legal_moves_text += f"Hound at {move_from} can move to:\n"
                    for move_to in moves_to:
                        legal_moves_text += f"  {move_id}. {move_to}\n"
                        move_id += 1
        return legal_moves_text

    @staticmethod
    def _last_opponent_move(board: Board, board_after_own_move: Board | None) -> Move | None:
        if board_after_own_move is None:
            if board.turn_count == 0:
                return None
            if board.turn_count != 1:
                return None
            previous = Board(board.size)
        else:
            previous = board_after_own_move

        for move in previous.get_legal_moves():
            candidate = previous.clone()
            if (
                candidate.make_move(move)
                and candidate.fox_pos == board.fox_pos
                and candidate.hounds_pos == board.hounds_pos
                and candidate.current_turn == board.current_turn
            ):
                return move
        return None

    @staticmethod
    def _board_state(
        board: Board,
        legal_moves_text: str,
        last_opponent_move: Move | None,
        *,
        include_last_opponent_move: bool,
    ) -> str:
        if last_opponent_move is None:
            last_move_text = "None (opening move)." if board.turn_count == 0 else "Unavailable."
        else:
            last_move_text = f"{last_opponent_move[0]} -> {last_opponent_move[1]}"
        last_move_line = f"LAST OPPONENT MOVE: {last_move_text}\n" if include_last_opponent_move else ""

        return f"""--- CURRENT BOARD STATE ---
TURN: {board.turn_count}
YOU PLAY AS: {board.current_turn.name}
FOX POSITION: {board.fox_pos}
HOUNDS POSITIONS: {board.hounds_pos}
{last_move_line}
LEGAL MOVES:
{legal_moves_text}

BOARD:
{board.to_ascii()}
"""

    def _request(self, messages: list[ChatCompletionMessageParam]) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model_cfg.name,
            messages=messages,
            temperature=self.model_cfg.conn.temperature,
            response_format={"type": "json_object"},
            timeout=self.model_cfg.conn.timeout,
        )

    def _response_content(self, messages: list[ChatCompletionMessageParam]) -> str:
        response = self._request(messages)
        if not (raw_content := response.choices[0].message.content):
            raise ValueError(f"{self.model_cfg.name} returned an invalid message shape")
        return raw_content

    @staticmethod
    def _json_retry_prompt() -> ChatCompletionMessageParam:
        return {
            "role": "user",
            "content": "The previous request failed to produce valid JSON. Return a complete JSON object as requested, "
            "using a very short analysis.",
        }

    @override
    def get_move(self, board: Board) -> Move:
        system_prompt = self._system_prompt(board)
        legal_moves = board.get_legal_moves()
        legal_moves_text = self._legal_moves_text(board, legal_moves)
        last_opponent_move = (
            self._last_opponent_move(board, self._board_after_own_move)
            if self.context == ContextMode.CONVERSATION
            else None
        )
        board_state = self._board_state(
            board,
            legal_moves_text,
            last_opponent_move,
            include_last_opponent_move=self.context == ContextMode.CONVERSATION,
        )
        messages: list[ChatCompletionMessageParam] = (
            self._conversation.copy()
            if self.context == ContextMode.CONVERSATION and self._conversation
            else [{"role": "system", "content": system_prompt}]
        )

        threat_analysis: str | None = None
        if self.reasoning == ReasoningMode.ANALYSIS:
            analysis_prompt = f"""Review the current position and identify the most important immediate threats and opportunities for your side.
Do not choose or return a move. Respond ONLY with a JSON object containing one field, "analysis", with a concise threat analysis.

{board_state}"""
            analysis_messages = messages + [{"role": "user", "content": analysis_prompt}]
            try:
                analysis_content = self._response_content(analysis_messages.copy())
            except BadRequestError as exc:
                if exc.code != "json_validate_failed":
                    raise
                analysis_messages.append(self._json_retry_prompt())
                try:
                    analysis_content = self._response_content(analysis_messages.copy())
                except BadRequestError as retry_exc:
                    if retry_exc.code == "json_validate_failed":
                        retry_exc.add_note("Threat analysis failed to generate valid JSON after one retry.")
                    raise
            threat_analysis = ThreatAnalysis.model_validate_json(analysis_content).analysis
            if self.context == ContextMode.CONVERSATION:
                messages = analysis_messages + [{"role": "assistant", "content": threat_analysis}]
        selection_prompt = f"""You MUST choose ONE legal move from the list below.
Respond ONLY with a JSON object matching this exact schema:
- move_id: 1-based integer ID of your chosen legal move
- analysis: short sentence explaining your move's strategic purpose, suitable for display

Choose only an ID listed below. Do not return move coordinates, additional fields, Markdown, or text outside the JSON object.
Valid IDs are 1 through {len(legal_moves)} inclusive.
Make the explanation specific to the chosen move and current board; avoid claiming a guaranteed win unless it follows directly."""
        if threat_analysis is not None:
            selection_prompt += f"""

THREAT ANALYSIS:
{threat_analysis}"""

        selection_prompt += f"\n\n{board_state}"

        turn_messages = messages + [{"role": "user", "content": selection_prompt}]
        for attempt in range(2):
            try:
                raw_content = self._response_content(turn_messages.copy())
            except BadRequestError as exc:
                if exc.code != "json_validate_failed":
                    raise
                if attempt:
                    exc.add_note("Move selection failed to generate valid JSON after one retry.")
                    raise
                turn_messages.append(self._json_retry_prompt())
                continue
            decision = Decision.model_validate_json(raw_content)
            if 1 <= decision.move_id <= len(legal_moves):
                break
            if attempt:
                raise ValueError(
                    f"{self.model_cfg.name} returned move_id {decision.move_id} after one retry; "
                    f"expected 1..{len(legal_moves)}"
                )
            turn_messages.extend(
                [
                    {"role": "assistant", "content": raw_content},
                    {
                        "role": "user",
                        "content": f"""Your previous response used invalid move_id {decision.move_id}.
Move IDs are 1-based. Choose one legal move with an ID from 1 through {len(legal_moves)} inclusive.
Return only the required JSON object, and choose an ID exactly as listed below.

LEGAL MOVES:
{legal_moves_text}""",
                    },
                ]
            )

        move = legal_moves[decision.move_id - 1]
        self.analysis = decision.analysis
        if self.context == ContextMode.CONVERSATION:
            turn_messages.append({"role": "assistant", "content": raw_content})
            self._conversation = turn_messages
            self._board_after_own_move = board.clone()
            if not self._board_after_own_move.make_move(move):
                raise ValueError(f"{self.model_cfg.name} selected a move that is no longer legal")
        return move
