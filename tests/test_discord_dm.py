from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from discord_bridge.bridge import VoiceBridge
from discord_bridge.config import BridgeConfig


class DiscordDmTests(unittest.TestCase):
    def test_resolves_configured_user_dm(self) -> None:
        cfg = BridgeConfig()
        cfg.reply.destination = "dm"
        cfg.reply.dm_user_id = 1234
        bridge = VoiceBridge(cfg)

        dm = object()
        user = Mock()
        user.create_dm = AsyncMock(return_value=dm)
        client = Mock()
        client.get_user.return_value = user

        result = asyncio.run(bridge._resolve_reply_destination(client))

        self.assertIs(result, dm)
        client.get_user.assert_called_once_with(1234)
        user.create_dm.assert_awaited_once()

    def test_missing_dm_user_does_not_open_dm(self) -> None:
        cfg = BridgeConfig()
        cfg.reply.destination = "dm"
        bridge = VoiceBridge(cfg)
        client = Mock()

        result = asyncio.run(bridge._resolve_reply_destination(client))

        self.assertIsNone(result)
        client.get_user.assert_not_called()

    def test_dm_without_configured_user_reaches_message_handler(self) -> None:
        cfg = BridgeConfig()
        cfg.reply.destination = "dm"
        bridge = VoiceBridge(cfg)
        bridge._handle_audio_attachment = AsyncMock(return_value=True)
        author = Mock(id=123456789012345678, bot=False)
        message = Mock(author=author, guild=None, channel=Mock(), id=1)
        client = Mock(user=Mock(id=999))

        asyncio.run(bridge._handle_message(client, message))

        bridge._handle_audio_attachment.assert_awaited_once_with(message)



if __name__ == "__main__":
    unittest.main()
