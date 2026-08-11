"""Bot から見えるチャンネルを全部並べる。

設定に入れる ID を調べるために使う。接続して一覧を取ったら切断する。
音声は触らないので、通話中でも安全に実行できる。

    python list_channels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discord_bridge.bridge import VoiceBridge  # noqa: E402
from discord_bridge.config import BridgeConfig  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "discord_bridge_config.json"


def main() -> int:
    cfg = BridgeConfig.load(CONFIG_PATH) if CONFIG_PATH.exists() else BridgeConfig()
    bridge = VoiceBridge(cfg, log=lambda m: print(m))

    print(f"トークンの環境変数: {cfg.token_env}")
    print(f".env: {cfg.env_file}")
    print("接続して一覧を取る…\n")

    rows = bridge.list_channels()
    if not rows:
        print("チャンネルが取れなかった。トークンと Bot の招待状況を確認してくれ。")
        return 1

    by_guild: dict[str, list[dict]] = {}
    for r in rows:
        by_guild.setdefault(r["guild"], []).append(r)

    for guild, items in by_guild.items():
        print(f"■ {guild}")
        for r in sorted(items, key=lambda x: (x["type"], x["name"])):
            kind = r["type"].replace("Channel", "")
            print(f"   {kind:<10} {r['name']:<28} {r['id']}")
        print()

    print("設定への入れ方:")
    print("  capture.voice_channel_id … 音声を拾う Voice チャンネルの ID（現在未実装）")
    print("  reply.text_channel_id    … テキストと画像を送る Text チャンネルの ID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
