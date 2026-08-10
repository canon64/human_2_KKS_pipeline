from __future__ import annotations

import unittest
from pathlib import Path

from discord_bridge.config import BridgeConfig


class DiscordConfigTests(unittest.TestCase):
    def test_default_env_is_next_to_tool(self) -> None:
        expected = Path(__file__).resolve().parents[1] / ".env"
        self.assertEqual(Path(BridgeConfig().env_file), expected)


if __name__ == "__main__":
    unittest.main()
