"""Discord ボイスブリッジを起動する。

    python run_bridge.py

ボイスチャンネルに入り、喋った区間を WAV にして
設定した監視フォルダへ落とす。
そこから先は既存のパイプラインがそのまま拾う。

停止は Ctrl+C。
"""

from __future__ import annotations

import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discord_bridge.bridge import VoiceBridge  # noqa: E402
from discord_bridge.config import BridgeConfig  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "discord_bridge_config.json"
LOG_PATH = Path(__file__).resolve().parents[1] / "discord_bridge.log"


def make_logger():
    """画面とファイルの両方へ。ファイルは起動のたびに空にする。"""
    LOG_PATH.write_text(
        f"=== discord_voice_bridge log start {datetime.now():%Y-%m-%d %H:%M:%S} ===\n",
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


def preflight(cfg: BridgeConfig, log) -> bool:
    """動かす前に、足りないものをここで全部出す。"""
    ok = True

    try:
        import discord

        log(f"py-cord: {discord.__version__}")
        if not discord.opus.is_loaded():
            try:
                discord.opus._load_default()
            except Exception:
                pass
        if discord.opus.is_loaded():
            log("opus: 読み込み済み")
        else:
            log("opus: 読み込めない → 音声の送受信ができない")
            ok = False
        if not hasattr(discord, "sinks"):
            log("sinks が無い → 音声受信ができない。py-cord[voice] が要る")
            ok = False
    except ImportError:
        log("py-cord が入っていない: pip install py-cord[voice]")
        return False

    if shutil.which("ffmpeg"):
        log("ffmpeg: あり")
    else:
        log("ffmpeg: 無し → 通話への音声再生ができない（受信は可能）")

    if not cfg.capture.voice_channel_id:
        # 通話に入らないだけで、テキストの往復には関係ない。
        # Discord の E2E 暗号化により音声受信は現状不可なので、0 が普通の設定。
        log("voice_channel_id が未設定 → 通話には入らない（テキストは動く）")

    out = Path(cfg.wav.output_dir) if cfg.wav.output_dir else None
    if out is None:
        log("wav.output_dir が未設定 → WAV を書き出せない（音声を使わないなら問題ない）")
    elif not out.exists():
        log(f"wav.output_dir が無い: {out}")
    else:
        log(f"wav 出力先: {out}")

    if cfg.reply.destination == "dm":
        if cfg.reply.dm_user_id:
            log(f"送受信先: 個人DM user={cfg.reply.dm_user_id}")
        elif cfg.reply.dm_username:
            log(f"送受信先: 個人DM username={cfg.reply.dm_username}（初回DMでIDを確認）")
        else:
            log("DMのユーザー名またはIDが未設定 → DMを送受信できない")
    elif not cfg.reply.text_channel_id:
        log("text_channel_id が未設定 → テキストと画像は送らない（受信のみ）")

    return ok


def kill_existing_bridges(log) -> int:
    """
    先に動いているブリッジを落とす。

    二重に起動すると、1つのメッセージを両方が受け取り、それぞれ Grok へ投げる。
    実際に PID 30684 と 15080 が並走し、同じ入力が2回入った。
    メッセージIDの重複除去はプロセス内でしか効かないので、ここで潰す。
    """
    import os
    import subprocess

    me = os.getpid()
    killed = 0
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*run_bridge*' } | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception as exc:
        log(f"既存プロセスの確認に失敗: {exc}")
        return 0

    for line in out.splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        pid = int(line)
        if pid == me:
            continue
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=10)
            log(f"既に動いていたブリッジを落とした pid={pid}")
            killed += 1
        except Exception as exc:
            log(f"pid={pid} を落とせない: {exc}")
    return killed


def main() -> int:
    log = make_logger()
    cfg = BridgeConfig.load(CONFIG_PATH)

    log(f"設定: {CONFIG_PATH}")
    kill_existing_bridges(log)
    if not preflight(cfg, log):
        log("前提が足りないので起動しない")
        return 1

    log(
        f"しきい値 rms={cfg.capture.rms_threshold} "
        f"無音={cfg.capture.silence_close_sec}s "
        f"最短={cfg.capture.min_utterance_sec}s "
        f"最長={cfg.capture.max_utterance_sec}s"
    )

    bridge = VoiceBridge(cfg, log=log)
    if not bridge.start():
        log("起動に失敗した")
        return 1

    stopping = False

    def _stop(signum, frame):  # noqa: ANN001, ARG001
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)

    log("動作中。Ctrl+C で止める")
    try:
        while not stopping:
            time.sleep(0.5)
    finally:
        log("停止処理…")
        bridge.flush_all()
        log("終了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
