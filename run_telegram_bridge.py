"""Telegram ブリッジを起動する。

    python run_telegram_bridge.py

Bot へ送ったテキストを Human_2_kks の /ask へ渡し、
Grok の返事と SD 画像を返す。音声メッセージは WAV へ落として監視フォルダへ置く。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from discord_bridge.pipeline_client import PipelineClient, PipelineConfig  # noqa: E402
from telegram_bridge.bridge import TelegramBridge, TelegramConfig, load_env_file  # noqa: E402

CONFIG_PATH = ROOT / "telegram_bridge_config.json"
LOG_PATH = ROOT / "telegram_bridge.log"


def make_logger():
    LOG_PATH.write_text(
        f"=== telegram_bridge log start {datetime.now():%Y-%m-%d %H:%M:%S} ===\n",
        encoding="utf-8",
    )

    def log(message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        print(line, flush=True)
        try:
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    return log


def main() -> int:
    log = make_logger()

    raw = {}
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    cfg = TelegramConfig(
        token_env=raw.get("token_env", "TELEGRAM_BOT_TOKEN"),
        env_file=raw.get("env_file", r"J:\tools\api-scripts\runtime\.env"),
        allowed_chat_ids=list(raw.get("allowed_chat_ids") or []),
        message_limit=int(raw.get("message_limit", 3900)),
        max_images=int(raw.get("max_images", 1)),
        accept_voice=bool(raw.get("accept_voice", True)),
        wav_output_dir=raw.get("wav_output_dir", r"F:\kks\wave"),
    )

    load_env_file(cfg.env_file)
    token = os.environ.get(cfg.token_env, "").strip()
    if not token:
        log(f"トークンが無い（環境変数 {cfg.token_env}）")
        return 1

    if shutil.which("ffmpeg"):
        log("ffmpeg: あり")
    else:
        log("ffmpeg: 無し → 音声メッセージを変換できない")

    pipe = PipelineClient(
        PipelineConfig(
            host=raw.get("pipeline_host", "127.0.0.1"),
            port=int(raw.get("pipeline_port", 18767)),
            token=raw.get("pipeline_token", ""),
            timeout_sec=float(raw.get("timeout_sec", 180)),
        ),
        log=log,
    )

    bridge = TelegramBridge(cfg, token, ask=pipe.ask, log=log)

    def save_chat_ids() -> None:
        raw["allowed_chat_ids"] = cfg.allowed_chat_ids
        raw.setdefault("token_env", cfg.token_env)
        raw.setdefault("env_file", cfg.env_file)
        raw.setdefault("wav_output_dir", cfg.wav_output_dir)
        raw.setdefault("pipeline_port", 18767)
        raw.setdefault("timeout_sec", 180)
        CONFIG_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        bridge.run()
    except KeyboardInterrupt:
        log("停止")
    finally:
        save_chat_ids()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
