from __future__ import annotations

import unittest
from pathlib import Path

from discord_bridge.config import BridgeConfig, parse_discord_id


class DiscordConfigTests(unittest.TestCase):
    def test_default_env_is_next_to_tool(self) -> None:
        expected = Path(__file__).resolve().parents[1] / ".env"
        self.assertEqual(Path(BridgeConfig().env_file), expected)

    def test_parses_discord_id_and_removes_mention_wrapper_only(self) -> None:
        expected = 123456789012345678
        self.assertEqual(parse_discord_id("<@123456789012345678>"), expected)
        self.assertEqual(parse_discord_id("<@!123456789012345678>"), expected)
        self.assertEqual(parse_discord_id("123456789012345678"), expected)

    def test_rejects_missing_or_short_id(self) -> None:
        self.assertEqual(parse_discord_id(""), 0)
        self.assertEqual(parse_discord_id("12345"), 0)
        self.assertEqual(parse_discord_id("labun0741"), 0)


if __name__ == "__main__":
    unittest.main()
