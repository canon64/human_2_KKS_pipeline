from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from grok_bridge.config import BridgeConfig, load_or_create_config
from grok_bridge.llm_providers import (
    LlmRequestConfig,
    compose_llm_input,
    generate_llm_response,
)
from grok_bridge.tts_event_cli import (
    _build_arg_parser,
    _limit_response_text,
    _should_use_live_grok_stream,
)


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def history_candidate(
    assistant_text: str = " 返答本文\nそのまま ",
    *,
    rank: int = 1,
    score: float = 0.8765,
    message_id: int = 101,
) -> dict[str, object]:
    return {
        "rank": rank,
        "score": score,
        "message_id": message_id,
        "conversation_id": f"conversation-{message_id}",
        "user_text": f"一致したユーザー本文-{message_id}",
        "assistant_text": assistant_text,
        "assistant_replies": [
            {
                "message_id": message_id + 1,
                "node_id": f"assistant-node-{message_id}",
                "text": assistant_text,
            }
        ],
    }


def history_payload(*candidates: dict[str, object]) -> dict[str, object]:
    results = list(candidates) or [history_candidate()]
    return {"ok": True, "selected": results[0], "results": results}


class GrokHistoryProviderTests(unittest.TestCase):
    def test_grok_browser_label_uses_history_api_and_returns_exact_assistant_body(self) -> None:
        logger = logging.getLogger("test_grok_history_provider")
        captured_request = None

        def fake_urlopen(request, timeout):
            nonlocal captured_request
            captured_request = request
            self.assertEqual(timeout, 7.0)
            return FakeHttpResponse(history_payload(history_candidate()))

        config = BridgeConfig(
            history_search_url="http://127.0.0.1:8877/search",
            history_search_timeout_seconds=7.0,
        )
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), self.assertLogs(
            logger, level="INFO"
        ) as logs:
            backend, response = generate_llm_response(
                "  質問本文だけ  ",
                llm_config=LlmRequestConfig(
                    backend="grok_browser",
                    grok_history_enabled=True,
                ),
                bridge_config=config,
                logger=logger,
            )

        self.assertEqual(backend, "grok_browser")
        self.assertEqual(response, " 返答本文\nそのまま ")
        self.assertIsNotNone(captured_request)
        request_body = json.loads(captured_request.data.decode("utf-8"))
        self.assertEqual(request_body, {"query": "質問本文だけ", "top_k": 10})
        self.assertEqual(captured_request.full_url, "http://127.0.0.1:8877/search")
        joined_logs = "\n".join(logs.output)
        self.assertIn("conversation-101", joined_logs)
        self.assertIn("assistant_text_chars", joined_logs)
        self.assertNotIn("一致したユーザー本文", joined_logs)
        self.assertNotIn("返答本文", joined_logs)

    def test_random_selection_uses_configured_top_k_candidate_pool(self) -> None:
        first = history_candidate("候補A", rank=1, score=0.9, message_id=201)
        second = history_candidate("候補B", rank=2, score=0.8, message_id=301)
        captured_request = None

        def fake_urlopen(request, timeout):
            nonlocal captured_request
            captured_request = request
            self.assertEqual(timeout, 12.0)
            return FakeHttpResponse(history_payload(first, second))

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), mock.patch(
            "grok_bridge.llm_providers.random.choice",
            side_effect=lambda pool: pool[-1],
        ):
            _, response = generate_llm_response(
                "質問",
                llm_config=LlmRequestConfig(
                    grok_history_top_k=10,
                    grok_history_selection_mode="random",
                    grok_history_timeout_seconds=12.0,
                ),
                bridge_config=BridgeConfig(),
                logger=logging.getLogger("test_grok_history_random"),
            )

        self.assertEqual(response, "候補B")
        self.assertIsNotNone(captured_request)
        request_body = json.loads(captured_request.data.decode("utf-8"))
        self.assertEqual(request_body["top_k"], 10)

    def test_required_terms_filter_and_preferred_terms_override_vector_rank(self) -> None:
        high_plain = history_candidate("普通の返答", rank=1, score=0.99, message_id=401)
        required_only = history_candidate("必須を含む返答", rank=2, score=0.92, message_id=501)
        preferred = history_candidate(
            "必須と優先を含む返答", rank=3, score=0.70, message_id=601
        )

        with mock.patch(
            "urllib.request.urlopen",
            return_value=FakeHttpResponse(
                history_payload(high_plain, required_only, preferred)
            ),
        ):
            _, response = generate_llm_response(
                "質問",
                llm_config=LlmRequestConfig(
                    grok_history_response_required_terms="必須",
                    grok_history_response_preferred_terms="優先",
                ),
                bridge_config=BridgeConfig(),
                logger=logging.getLogger("test_grok_history_terms"),
            )

        self.assertEqual(response, "必須と優先を含む返答")

    def test_required_all_mode_excludes_partial_matches(self) -> None:
        partial = history_candidate("必須だけ", rank=1, score=0.9, message_id=701)
        complete = history_candidate("必須と追加の両方", rank=2, score=0.8, message_id=801)

        with mock.patch(
            "urllib.request.urlopen",
            return_value=FakeHttpResponse(history_payload(partial, complete)),
        ):
            _, response = generate_llm_response(
                "質問",
                llm_config=LlmRequestConfig(
                    grok_history_required_match_mode="all",
                    grok_history_response_required_terms="必須\n追加",
                ),
                bridge_config=BridgeConfig(),
                logger=logging.getLogger("test_grok_history_required_all"),
            )

        self.assertEqual(response, "必須と追加の両方")

    def test_random_selection_prefers_highest_keyword_match_group(self) -> None:
        no_preference = history_candidate("通常", rank=1, score=0.99, message_id=901)
        preferred = history_candidate("優先を含む", rank=2, score=0.70, message_id=1001)

        with mock.patch(
            "urllib.request.urlopen",
            return_value=FakeHttpResponse(history_payload(no_preference, preferred)),
        ), mock.patch(
            "grok_bridge.llm_providers.random.choice",
            side_effect=lambda pool: pool[0],
        ) as choose:
            _, response = generate_llm_response(
                "質問",
                llm_config=LlmRequestConfig(
                    grok_history_selection_mode="random",
                    grok_history_response_preferred_terms="優先",
                ),
                bridge_config=BridgeConfig(),
                logger=logging.getLogger("test_grok_history_random_preferred"),
            )

        self.assertEqual(response, "優先を含む")
        self.assertEqual(len(choose.call_args.args[0]), 1)

    def test_min_score_no_match_can_fallback_to_live_grok(self) -> None:
        low_score = history_candidate("低スコア", score=0.2, message_id=1101)
        with mock.patch(
            "urllib.request.urlopen",
            return_value=FakeHttpResponse(history_payload(low_score)),
        ), mock.patch(
            "grok_bridge.llm_providers._generate_grok_live_response",
            return_value="ライブ応答",
        ) as live:
            _, response = generate_llm_response(
                "質問",
                llm_config=LlmRequestConfig(
                    grok_history_min_score=0.5,
                    grok_history_fallback_live=True,
                ),
                bridge_config=BridgeConfig(),
                logger=logging.getLogger("test_grok_history_fallback"),
            )

        self.assertEqual(response, "ライブ応答")
        live.assert_called_once()

    def test_history_response_enters_existing_downstream_without_provider_rewrite(self) -> None:
        response = "保存済みassistant本文\n二行目"
        limited, raw_len, capped_len, truncated = _limit_response_text(
            response,
            max_chars=0,
            logger=logging.getLogger("test_grok_history_downstream"),
            source="grok_browser",
        )
        self.assertEqual(limited, response)
        self.assertEqual(raw_len, len(response))
        self.assertEqual(capped_len, len(response))
        self.assertFalse(truncated)

    def test_history_off_uses_live_grok_without_calling_history_api(self) -> None:
        config = BridgeConfig()
        with mock.patch(
            "grok_bridge.llm_providers._generate_grok_live_response",
            return_value="ライブGrok応答",
        ) as live, mock.patch(
            "grok_bridge.llm_providers._generate_grok_history_response"
        ) as history:
            backend, response = generate_llm_response(
                "ライブへ送る質問",
                llm_config=LlmRequestConfig(
                    backend="grok_browser",
                    grok_history_enabled=False,
                ),
                bridge_config=config,
                logger=logging.getLogger("test_grok_live_provider"),
            )
        self.assertEqual(backend, "grok_browser")
        self.assertEqual(response, "ライブGrok応答")
        live.assert_called_once()
        history.assert_not_called()

    def test_cli_history_switch_defaults_on_and_can_turn_off(self) -> None:
        parser = _build_arg_parser()
        self.assertTrue(parser.parse_args([]).grok_history)
        self.assertTrue(parser.parse_args(["--grok-history"]).grok_history)
        self.assertFalse(parser.parse_args(["--no-grok-history"]).grok_history)

    def test_cli_accepts_vector_response_selection_and_filter_settings(self) -> None:
        args = _build_arg_parser().parse_args(
            [
                "--grok-history-top-k",
                "10",
                "--grok-history-selection-mode",
                "random",
                "--grok-history-min-score",
                "0.35",
                "--grok-history-fallback-live",
                "--grok-history-required-match-mode",
                "all",
                "--grok-history-response-required-terms",
                "必須\n追加",
                "--grok-history-response-preferred-terms",
                "優先",
                "--llm-always-append-text",
                "固定語",
            ]
        )
        self.assertEqual(args.grok_history_top_k, 10)
        self.assertEqual(args.grok_history_selection_mode, "random")
        self.assertEqual(args.grok_history_min_score, 0.35)
        self.assertTrue(args.grok_history_fallback_live)
        self.assertEqual(args.grok_history_required_match_mode, "all")
        self.assertEqual(args.grok_history_response_required_terms, "必須\n追加")
        self.assertEqual(args.grok_history_response_preferred_terms, "優先")
        self.assertEqual(args.llm_always_append_text, "固定語")

    def test_fixed_text_is_appended_only_to_llm_input(self) -> None:
        original = "ユーザー発言"
        self.assertEqual(compose_llm_input(original, "必ず送る語"), "ユーザー発言 必ず送る語")
        self.assertEqual(compose_llm_input(original, ""), original)
        self.assertEqual(compose_llm_input("", "固定語"), "固定語")

    def test_live_stream_is_only_enabled_when_history_is_off(self) -> None:
        parser = _build_arg_parser()
        common = ["--stream", "--sbv2-server-url", "http://127.0.0.1:5000", "--text", "質問"]
        self.assertFalse(_should_use_live_grok_stream(parser.parse_args(common)))
        self.assertTrue(
            _should_use_live_grok_stream(parser.parse_args([*common, "--no-grok-history"]))
        )

    def test_history_api_must_be_loopback(self) -> None:
        config = BridgeConfig(history_search_url="https://example.com/search")
        with self.assertRaisesRegex(RuntimeError, "loopback"):
            generate_llm_response(
                "質問",
                llm_config=LlmRequestConfig(backend="grok_browser"),
                bridge_config=config,
                logger=logging.getLogger("test_grok_history_loopback"),
            )

    def test_old_bridge_config_gets_history_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "grok_bridge_config.json"
            path.write_text("{}\n", encoding="utf-8")
            config = load_or_create_config(str(path))
        self.assertEqual(config.history_search_url, "http://127.0.0.1:8877/search")
        self.assertEqual(config.history_search_timeout_seconds, 30.0)


if __name__ == "__main__":
    unittest.main()
