from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_bridge.token_store import save_env_secret


class DiscordTokenStoreTests(unittest.TestCase):
    def test_creates_and_updates_only_requested_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text("OTHER=value\nDISCORD_TOKEN=old\n", encoding="utf-8")

            save_env_secret(str(path), "DISCORD_TOKEN", "new-secret")

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "OTHER=value\nDISCORD_TOKEN=new-secret\n",
            )

    def test_rejects_empty_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "空"):
                save_env_secret(str(Path(temp_dir) / ".env"), "DISCORD_TOKEN", "")


if __name__ == "__main__":
    unittest.main()
