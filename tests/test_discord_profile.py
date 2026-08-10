from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from discord_bridge.profile import update_bot_profile


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return b'{"id":"1","username":"New Bot"}'


class DiscordProfileTests(unittest.TestCase):
    def test_updates_username_avatar_and_banner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_file = root / ".env"
            avatar = root / "avatar.png"
            banner = root / "banner.jpg"
            env_file.write_text("TEST_DISCORD_TOKEN=secret-token\n", encoding="utf-8")
            avatar.write_bytes(b"\x89PNG\r\n\x1a\nimage")
            banner.write_bytes(b"\xff\xd8\xffimage")

            with patch("discord_bridge.profile.urllib.request.urlopen", return_value=_Response()) as mocked:
                result = update_bot_profile(
                    token_env="TEST_DISCORD_TOKEN",
                    env_file=str(env_file),
                    username="New Bot",
                    avatar_path=str(avatar),
                    banner_path=str(banner),
                )

            request = mocked.call_args.args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["username"], "New Bot")
            self.assertTrue(payload["avatar"].startswith("data:image/png;base64,"))
            self.assertTrue(payload["banner"].startswith("data:image/jpeg;base64,"))
            self.assertEqual(request.headers["Authorization"], "Bot secret-token")
            self.assertEqual(result["username"], "New Bot")

    def test_rejects_empty_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("TEST_DISCORD_TOKEN=secret-token\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Bot名、アイコン画像、バナー画像"):
                update_bot_profile(
                    token_env="TEST_DISCORD_TOKEN",
                    env_file=str(env_file),
                )

    def test_rejects_invalid_username_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("TEST_DISCORD_TOKEN=secret-token\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "2～32文字"):
                update_bot_profile(
                    token_env="TEST_DISCORD_TOKEN",
                    env_file=str(env_file),
                    username="x",
                )


if __name__ == "__main__":
    unittest.main()
