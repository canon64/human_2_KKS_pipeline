"""新しい Bot へ切り替える。

    python discord_bridge/switch_bot.py DISCORD_PRIVATE_BOT_TOKEN

やること:
  1. 指定した環境変数からトークンを読む
  2. その Bot が見えるサーバーとチャンネルを一覧する
  3. 設定へ token_env を書き、許可サーバーを絞る候補を出す

チャンネルIDは一覧を見て手で選ぶ。誤爆を避けるため自動では決めない。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from discord_bridge.bridge import VoiceBridge, load_env_file  # noqa: E402
from discord_bridge.config import BridgeConfig  # noqa: E402

CONFIG = ROOT / "discord_bridge_config.json"


def main() -> int:
    token_env = sys.argv[1] if len(sys.argv) > 1 else "DISCORD_PRIVATE_BOT_TOKEN"

    cfg = BridgeConfig.load(CONFIG)
    cfg.token_env = token_env

    load_env_file(cfg.env_file)
    if not os.environ.get(token_env, "").strip():
        print(f"環境変数 {token_env} が空だ。")
        print(f"  {cfg.env_file} に次の行を足す:")
        print(f"  {token_env}=（開発者ポータルで取ったトークン）")
        return 1

    bridge = VoiceBridge(cfg, log=print)
    rows = bridge.list_channels()
    if not rows:
        print("チャンネルが取れない。Bot をサーバーへ招待したか確認する。")
        return 1

    by_guild: dict[str, list[dict]] = {}
    for r in rows:
        by_guild.setdefault(f"{r['guild']} ({r['guild_id']})", []).append(r)

    for guild, items in by_guild.items():
        print(f"\n■ {guild}")
        for r in sorted(items, key=lambda x: (x["type"], x["name"])):
            print(f"   {r['type'].replace('Channel',''):<10} {r['name']:<28} {r['id']}")

    # token_env だけ先に確定させる
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["token_env"] = token_env
    CONFIG.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ntoken_env を {token_env} にした。")
    print("上の一覧から、使うサーバーIDとチャンネルIDを GUI の Discord タブへ入れる。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
