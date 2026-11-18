import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx2
from openai import BadRequestError, OpenAI

from board import Board
from config import ConnectionConfig, ModelConfig
from players import ContextMode, LLMPlayer, ReasoningMode


def _reply(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="test-model",
        conn=ConnectionConfig(api_key="test-key"),
    )


def _bad_request(code: str) -> BadRequestError:
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://example.test"))
    return BadRequestError("Bad request", response=response, body={"code": code})


class LLMPlayerTests(unittest.TestCase):
    def test_default_is_direct_stateless_and_uses_one_based_move_ids(self):
        completions = Mock()
        completions.create.return_value = _reply(json.dumps({"move_id": 1, "analysis": "Advance."}))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("players.llm_player.OpenAI", return_value=client):
            player = LLMPlayer(_model_config())

        board = Board(8)
        move = player.get_move(board)

        self.assertEqual(move, board.get_legal_moves()[0])
        self.assertEqual(completions.create.call_count, 1)
        self.assertEqual(player.context, ContextMode.STATELESS)
        self.assertEqual(player.reasoning, ReasoningMode.DIRECT)
        self.assertNotIn("LAST OPPONENT MOVE", completions.create.call_args.kwargs["messages"][-1]["content"])
        self.assertNotIn("reasoning_effort", completions.create.call_args.kwargs)
        self.assertIn("1-based integer ID", completions.create.call_args.kwargs["messages"][-1]["content"])
        self.assertIn("Valid IDs are 1 through 1 inclusive", completions.create.call_args.kwargs["messages"][-1]["content"])

    def test_analysis_mode_uses_a_separate_analysis_call(self):
        completions = Mock()
        completions.create.side_effect = [
            _reply(json.dumps({"analysis": "The fox can advance safely."})),
            _reply(json.dumps({"move_id": 1, "analysis": "Advance."})),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("players.llm_player.OpenAI", return_value=client):
            player = LLMPlayer(_model_config(), reasoning=ReasoningMode.ANALYSIS)

        move = player.get_move(Board(8))

        self.assertEqual(move, Board(8).get_legal_moves()[0])
        self.assertEqual(completions.create.call_count, 2)
        for call in completions.create.call_args_list:
            self.assertNotIn("reasoning_effort", call.kwargs)
        selection_prompt = completions.create.call_args_list[1].kwargs["messages"][-1]["content"]
        self.assertIn("The fox can advance safely.", selection_prompt)

    def test_direct_conversation_reuses_turn_history(self):
        completions = Mock()
        completions.create.side_effect = [
            _reply(json.dumps({"move_id": 1, "analysis": "First move."})),
            _reply(json.dumps({"move_id": 1, "analysis": "Second move."})),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("players.llm_player.OpenAI", return_value=client):
            player = LLMPlayer(_model_config(), context=ContextMode.CONVERSATION)

        board = Board(8)
        own_move = player.get_move(board)
        self.assertTrue(board.make_move(own_move))
        self.assertTrue(board.make_move(board.get_legal_moves()[0]))
        player.get_move(board)

        second_messages = completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(len(second_messages), 4)
        self.assertIn("First move.", str(second_messages))
        self.assertIn(f"TURN: {board.turn_count}", second_messages[-1]["content"])

    def test_conversation_includes_previous_opponent_move_and_is_per_instance(self):
        completions = Mock()
        completions.create.side_effect = [
            _reply(json.dumps({"analysis": "First threat review."})),
            _reply(json.dumps({"move_id": 1, "analysis": "First move."})),
            _reply(json.dumps({"analysis": "Second threat review."})),
            _reply(json.dumps({"move_id": 1, "analysis": "Second move."})),
            _reply(json.dumps({"move_id": 1, "analysis": "Fresh instance."})),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("players.llm_player.OpenAI", return_value=client):
            player = LLMPlayer(
                _model_config(),
                reasoning=ReasoningMode.ANALYSIS,
                context=ContextMode.CONVERSATION,
            )
            fresh_player = LLMPlayer(_model_config(), context=ContextMode.CONVERSATION)

        board = Board(8)
        own_move = player.get_move(board)
        self.assertTrue(board.make_move(own_move))
        opponent_move = board.get_legal_moves()[0]
        self.assertTrue(board.make_move(opponent_move))
        player.get_move(board)

        second_analysis_messages = completions.create.call_args_list[2].kwargs["messages"]
        second_analysis_prompt = second_analysis_messages[-1]["content"]
        self.assertIn(f"{opponent_move[0]} -> {opponent_move[1]}", second_analysis_prompt)
        self.assertIn("First threat review.", str(second_analysis_messages))

        fresh_player.get_move(Board(8))
        fresh_messages = completions.create.call_args_list[4].kwargs["messages"]
        self.assertEqual(len(fresh_messages), 2)

    def test_invalid_move_id_retries_with_corrected_range(self):
        board = Board(8)
        completions = Mock()
        completions.create.side_effect = [
            _reply(json.dumps({"move_id": 0, "analysis": "Invalid."})),
            _reply(json.dumps({"move_id": 1, "analysis": "Advance."})),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("players.llm_player.OpenAI", return_value=client):
            player = LLMPlayer(_model_config())

        move = player.get_move(board)

        self.assertEqual(move, board.get_legal_moves()[0])
        self.assertEqual(completions.create.call_count, 2)
        correction_prompt = completions.create.call_args_list[1].kwargs["messages"][-1]["content"]
        self.assertIn(f"1 through {len(board.get_legal_moves())}", correction_prompt)
        self.assertIn("Move IDs are 1-based", correction_prompt)

    def test_repeated_invalid_move_ids_fail_with_range(self):
        legal_move_count = len(Board(8).get_legal_moves())
        completions = Mock()
        completions.create.side_effect = [
            _reply(json.dumps({"move_id": 0, "analysis": "Invalid."})),
            _reply(json.dumps({"move_id": legal_move_count + 1, "analysis": "Still invalid."})),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("players.llm_player.OpenAI", return_value=client):
            player = LLMPlayer(_model_config())

        with self.assertRaisesRegex(
            ValueError,
            f"returned move_id {legal_move_count + 1} after one retry; expected 1\\.\\.{legal_move_count}",
        ):
            player.get_move(Board(8))
        self.assertEqual(completions.create.call_count, 2)

    def test_conversation_retains_retry_exchange(self):
        completions = Mock()
        completions.create.side_effect = [
            _reply(json.dumps({"move_id": 0, "analysis": "Invalid."})),
            _reply(json.dumps({"move_id": 1, "analysis": "Corrected."})),
            _reply(json.dumps({"move_id": 1, "analysis": "Next turn."})),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("players.llm_player.OpenAI", return_value=client):
            player = LLMPlayer(_model_config(), context=ContextMode.CONVERSATION)

        board = Board(8)
        self.assertTrue(board.make_move(player.get_move(board)))
        self.assertTrue(board.make_move(board.get_legal_moves()[0]))
        player.get_move(board)

        next_turn_messages = completions.create.call_args_list[2].kwargs["messages"]
        self.assertIn('"move_id": 0', str(next_turn_messages))
        self.assertIn("Your previous response used invalid move_id 0", str(next_turn_messages))

    def test_json_failure_retries_both_stages_and_preserves_history(self):
        completions = Mock()
        completions.create.side_effect = [
            _bad_request("json_validate_failed"),
            _reply(json.dumps({"analysis": "Watch the hounds."})),
            _bad_request("json_validate_failed"),
            _reply(json.dumps({"move_id": 1, "analysis": "Advance."})),
            _reply(json.dumps({"analysis": "Watch the hounds again."})),
            _reply(json.dumps({"move_id": 1, "analysis": "Advance again."})),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("players.llm_player.OpenAI", return_value=client):
            player = LLMPlayer(_model_config(), reasoning=ReasoningMode.ANALYSIS, context=ContextMode.CONVERSATION)

        board = Board(8)
        self.assertTrue(board.make_move(player.get_move(board)))
        self.assertTrue(board.make_move(board.get_legal_moves()[0]))
        player.get_move(board)

        analysis_retry = completions.create.call_args_list[1].kwargs["messages"]
        selection_retry = completions.create.call_args_list[3].kwargs["messages"]
        next_turn = completions.create.call_args_list[4].kwargs["messages"]
        for messages in (analysis_retry, selection_retry):
            self.assertIn("failed to produce valid JSON", messages[-1]["content"])
        self.assertEqual(sum("failed to produce valid JSON" in str(message) for message in next_turn), 2)
        self.assertEqual(len([message for message in next_turn if message["role"] == "assistant"]), 2)

    def test_openai_client_retries_provider_json_validation_error(self):
        requests = []

        def respond(request: httpx2.Request) -> httpx2.Response:
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return httpx2.Response(
                    400,
                    json={
                        "error": {
                            "message": "Failed to generate JSON",
                            "code": "json_validate_failed",
                            "failed_generation": "incomplete secret generation",
                        }
                    },
                )
            return httpx2.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"move_id": 1, "analysis": "Advance."}),
                            },
                        }
                    ],
                },
            )

        with httpx2.Client(transport=httpx2.MockTransport(respond)) as http_client:
            client = OpenAI(api_key="test-key", base_url="https://example.test/v1", http_client=http_client)
            with patch("players.llm_player.OpenAI", return_value=client):
                player = LLMPlayer(_model_config())
            board = Board(8)
            self.assertEqual(player.get_move(board), board.get_legal_moves()[0])

        self.assertEqual(len(requests), 2)
        self.assertNotIn("incomplete secret generation", json.dumps(requests[1]))
        self.assertIn("failed to produce valid JSON", requests[1]["messages"][-1]["content"])

    def test_repeated_json_failures_are_bounded_in_both_stages(self):
        for mode in (ReasoningMode.DIRECT, ReasoningMode.ANALYSIS):
            with self.subTest(mode=mode):
                last_error = _bad_request("json_validate_failed")
                completions = Mock()
                completions.create.side_effect = [_bad_request("json_validate_failed"), last_error]
                client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
                with patch("players.llm_player.OpenAI", return_value=client):
                    player = LLMPlayer(_model_config(), reasoning=mode, context=ContextMode.CONVERSATION)

                with self.assertRaises(BadRequestError) as raised:
                    player.get_move(Board(8))
                self.assertIs(raised.exception, last_error)
                self.assertIn("after one retry", raised.exception.__notes__[0])
                self.assertEqual(completions.create.call_count, 2)
                self.assertEqual(player._conversation, [])

    def test_unrelated_bad_requests_are_not_retried(self):
        for mode in (ReasoningMode.DIRECT, ReasoningMode.ANALYSIS):
            with self.subTest(mode=mode):
                error = _bad_request("invalid_request_error")
                completions = Mock()
                completions.create.side_effect = error
                client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
                with patch("players.llm_player.OpenAI", return_value=client):
                    player = LLMPlayer(_model_config(), reasoning=mode)
                with self.assertRaises(BadRequestError) as raised:
                    player.get_move(Board(8))
                self.assertIs(raised.exception, error)
                self.assertEqual(completions.create.call_count, 1)

    def test_selection_shares_retry_budget_between_json_and_invalid_id(self):
        invalid = _reply(json.dumps({"move_id": 0, "analysis": "Invalid."}))
        for json_first in (True, False):
            with self.subTest(json_first=json_first):
                error = _bad_request("json_validate_failed")
                completions = Mock()
                completions.create.side_effect = [error, invalid] if json_first else [invalid, error]
                client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
                with patch("players.llm_player.OpenAI", return_value=client):
                    player = LLMPlayer(_model_config())
                if json_first:
                    with self.assertRaisesRegex(ValueError, "move_id 0 after one retry"):
                        player.get_move(Board(8))
                else:
                    with self.assertRaises(BadRequestError) as raised:
                        player.get_move(Board(8))
                    self.assertIs(raised.exception, error)
                    self.assertIn("after one retry", raised.exception.__notes__[0])
                self.assertEqual(completions.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
