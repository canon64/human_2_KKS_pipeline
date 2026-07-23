from __future__ import annotations

import unittest

from grok_bridge.tts_event_cli import _build_arg_parser, _split_response_lines


class TtsLineFilterTests(unittest.TestCase):
    def test_symbol_only_lines_are_skipped(self) -> None:
        response = "こんにちは❤\n---\n**\n❤❤❤\n……\n123\nEnglish!"

        self.assertEqual(
            _split_response_lines(response),
            ["こんにちは❤", "123", "English!"],
        )

    def test_symbol_only_response_produces_no_tts_lines(self) -> None:
        self.assertEqual(_split_response_lines("---\n❤❤\n……"), [])

    def test_japanese_text_with_symbols_is_kept(self) -> None:
        response = "**返答です**❤\n【次の行】"

        self.assertEqual(
            _split_response_lines(response, target_chars=280),
            ["**返答です**❤", "【次の行】"],
        )

    def test_last_period_before_target_is_preferred(self) -> None:
        response = ("あ" * 49) + "。" + ("い" * 19) + "。" + ("う" * 20)

        lines = _split_response_lines(response, target_chars=80)

        self.assertEqual([len(line) for line in lines], [70, 20])
        self.assertTrue(lines[0].endswith("。"))

    def test_later_exclamation_or_heart_beats_an_earlier_period(self) -> None:
        for boundary in ("！", "!", "♡", "❤", "❤️", "💕", "🩷"):
            with self.subTest(boundary=boundary):
                response = ("あ" * 49) + "。" + ("い" * 19) + boundary + ("う" * 20)

                lines = _split_response_lines(response, target_chars=80)

                self.assertEqual(lines[0], response[: 69 + len(boundary)])
                self.assertTrue(lines[0].endswith(boundary))

    def test_heart_variants_are_complete_primary_breaks(self) -> None:
        variants = (
            "♡",
            "♡̆̈",
            "♥",
            "❤",
            "❤️",
            "❣️",
            "💕",
            "💖",
            "💔",
            "🖤",
            "🤍",
            "🤎",
            "🧡",
            "🩷",
            "🩵",
            "🩶",
            "🫶",
            "🫰",
            "🥰",
            "😍",
            "😻",
            "❤️‍🔥",
            "❤️‍🩹",
            "ღ",
            "ෆ",
            "ᰔ",
            "ᥫ᭡",
            "ꨄ",
            "ᡣ𐭩",
        )
        for heart in variants:
            with self.subTest(heart=heart):
                response = ("あ" * 69) + heart + ("い" * 20)

                lines = _split_response_lines(response, target_chars=80)

                self.assertEqual("".join(lines), response)
                self.assertTrue(lines[0].endswith(heart))
                self.assertEqual(lines[0], ("あ" * 69) + heart)

    def test_last_comma_is_used_when_period_is_absent(self) -> None:
        response = ("あ" * 49) + "、" + ("い" * 19) + "、" + ("う" * 20)

        lines = _split_response_lines(response, target_chars=80)

        self.assertEqual([len(line) for line in lines], [70, 20])
        self.assertTrue(lines[0].endswith("、"))

    def test_period_has_priority_over_a_later_comma(self) -> None:
        response = ("あ" * 49) + "。" + ("い" * 19) + "、" + ("う" * 20)

        lines = _split_response_lines(response, target_chars=80)

        self.assertEqual([len(line) for line in lines], [50, 40])
        self.assertTrue(lines[0].endswith("。"))

    def test_target_position_is_used_without_japanese_punctuation(self) -> None:
        lines = _split_response_lines("あ" * 165, target_chars=80)

        self.assertEqual([len(line) for line in lines], [80, 80, 5])

    def test_cli_exposes_line_break_target_setting(self) -> None:
        parser = _build_arg_parser()

        self.assertEqual(parser.parse_args([]).line_break_target_chars, 80)
        self.assertEqual(
            parser.parse_args(
                ["--line-break-target-chars", "72"]
            ).line_break_target_chars,
            72,
        )
        self.assertNotIn("--max-line-chars", parser.format_help())


if __name__ == "__main__":
    unittest.main()
