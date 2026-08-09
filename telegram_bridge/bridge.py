"""Telegram との接続部。

Discord 版から入れ替えたのは受信部だけ。/ask を叩く pipeline_client と、
テキスト分割の responder はそのまま流用する。

Discord に対して有利な点:
  - ボイスメッセージが添付ファイルとして届く。通話の暗号化(DAVE)に阻まれない
  - 既に発話単位で区切られているので、区間の切り出し(segmenter)が要らない
  - Bot と1対1なので、他の用途と混ざる事故が起きない

依存は標準ライブラリのみ。Telegram の Bot API は素の HTTP で叩ける。
"""

from __future__ import annotations

import json
import mimetypes
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

API_ROOT = "https://api.telegram.org"


@dataclass
class TelegramConfig:
    token_env: str = "TELEGRAM_BOT_TOKEN"
    env_file: str = r"J:\tools\api-scripts\runtime\.env"

    # このチャットだけ相手にする。空なら最初に話しかけてきた相手を覚える。
    # 他人が Bot を見つけても反応させないための歯止め。
    allowed_chat_ids: list[int] = field(default_factory=list)

    # 受信の待ち時間(秒)。long polling なので長くても負荷は上がらない。
    poll_timeout_sec: int = 25

    # Telegram の1メッセージ上限は 4096。余裕を持たせる。
    message_limit: int = 3900

    # 返信に付ける画像の枚数。
    max_images: int = 1

    # 音声メッセージを文字起こしへ回すか。
    accept_voice: bool = True

    # 変換後の WAV を置く場所。空ならテンポラリへ。
    wav_output_dir: str = ""


def load_env_file(path: str) -> int:
    p = Path(path)
    if not path or not p.exists():
        return 0
    import os

    loaded = 0
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def split_message(text: str, limit: int) -> list[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]
    out: list[str] = []
    rest = normalized
    while rest:
        if len(rest) <= limit:
            out.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut <= 0:
            cut = rest.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
    return [c for c in out if c]


class TelegramBridge:
    def __init__(
        self,
        config: TelegramConfig,
        token: str,
        *,
        ask: Callable[[str], Any],
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.token = token
        self.ask = ask
        self.log = log or (lambda m: print(m))
        self._offset = 0
        self._stop = False
        self._seen: list[int] = []

    # ------------------------------------------------------------------
    def _api(self, method: str, payload: dict | None = None, timeout: float = 40.0) -> dict:
        url = f"{API_ROOT}/bot{self.token}/{method}"
        data = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.loads(res.read().decode())
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:200]
            except Exception:
                pass
            self.log(f"[tg] HTTP {e.code} {method}: {body}")
            return {"ok": False}
        except Exception as exc:
            self.log(f"[tg] {method} 失敗: {exc}")
            return {"ok": False}

    def send_text(self, chat_id: int, text: str) -> None:
        for chunk in split_message(text, self.config.message_limit):
            self._api("sendMessage", {"chat_id": chat_id, "text": chunk})

    def send_photo(self, chat_id: int, image: bytes, caption: str = "") -> bool:
        """multipart で写真を送る。標準ライブラリだけで組む。"""
        boundary = "----" + uuid.uuid4().hex
        parts: list[bytes] = []

        def field(name: str, value: str) -> None:
            parts.append(
                (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                 f"{value}\r\n").encode("utf-8")
            )

        field("chat_id", str(chat_id))
        if caption:
            field("caption", caption[:1000])
        parts.append(
            (f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; "
             f"filename=\"sd.png\"\r\nContent-Type: image/png\r\n\r\n").encode("utf-8")
        )
        parts.append(image)
        parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(parts)

        req = urllib.request.Request(
            f"{API_ROOT}/bot{self.token}/sendPhoto", data=body, method="POST"
        )
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                return json.loads(res.read().decode()).get("ok", False)
        except Exception as exc:
            self.log(f"[tg] 画像の送信に失敗: {exc}")
            return False

    # ------------------------------------------------------------------
    def _download_file(self, file_id: str) -> Path | None:
        info = self._api("getFile", {"file_id": file_id})
        if not info.get("ok"):
            return None
        path = info["result"].get("file_path")
        if not path:
            return None
        url = f"{API_ROOT}/file/bot{self.token}/{path}"
        suffix = Path(path).suffix or ".oga"
        dst = Path(tempfile.gettempdir()) / f"tg_{uuid.uuid4().hex}{suffix}"
        try:
            with urllib.request.urlopen(url, timeout=60) as res:
                dst.write_bytes(res.read())
            return dst
        except Exception as exc:
            self.log(f"[tg] ダウンロード失敗: {exc}")
            return None

    def _to_wav(self, src: Path) -> Path | None:
        """faster-whisper が扱いやすい 16kHz モノラルへ落とす。"""
        out_dir = Path(self.config.wav_output_dir) if self.config.wav_output_dir else Path(tempfile.gettempdir())
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dst = out_dir / f"telegram_{stamp}_{uuid.uuid4().hex[:6]}.wav"
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
                capture_output=True, timeout=120,
            )
            if proc.returncode != 0 or not dst.exists():
                self.log(f"[tg] ffmpeg 変換に失敗: {proc.stderr.decode(errors='replace')[:160]}")
                return None
            return dst
        except Exception as exc:
            self.log(f"[tg] ffmpeg 実行に失敗: {exc}")
            return None

    # ------------------------------------------------------------------
    def _allowed(self, chat_id: int) -> bool:
        allow = self.config.allowed_chat_ids
        if not allow:
            # 最初に話しかけてきた相手を覚える。以後はその人だけ。
            self.config.allowed_chat_ids.append(chat_id)
            self.log(f"[tg] 相手を覚えた chat_id={chat_id}")
            return True
        return chat_id in allow

    def _handle(self, message: dict) -> None:
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if not chat_id or not self._allowed(int(chat_id)):
            return

        mid = message.get("message_id")
        if mid in self._seen:
            return
        self._seen.append(mid)
        if len(self._seen) > 200:
            del self._seen[:100]

        text = (message.get("text") or "").strip()

        # ボイスメッセージ / 音声ファイル
        voice = message.get("voice") or message.get("audio")
        if voice and self.config.accept_voice and not text:
            self.send_text(int(chat_id), "（音声を受け取った。文字起こし中…）")
            src = self._download_file(voice.get("file_id", ""))
            if src is None:
                self.send_text(int(chat_id), "（音声を取得できなかった）")
                return
            wav = self._to_wav(src)
            try:
                src.unlink(missing_ok=True)
            except Exception:
                pass
            if wav is None:
                self.send_text(int(chat_id), "（音声を変換できなかった）")
                return
            self.log(f"[tg] 音声 -> {wav.name}")
            # 監視フォルダへ置いた場合は、パイプラインが拾って文字起こしする。
            # ここでは返事を返すため、テキスト化を待たずに知らせるだけに留める。
            self.send_text(int(chat_id), f"（{wav.name} として渡した）")
            return

        if not text:
            return

        # Grok の入力欄は改行が送信になるので1行へ潰す
        if "\n" in text or "\r" in text:
            norm = text.replace("\r\n", "\n").replace("\r", "\n")
            text = " ".join(p.strip() for p in norm.split("\n") if p.strip())

        self.log(f"[tg] 受信: {text[:60]}")
        result = self.ask(text)

        if result is None or not getattr(result, "ok", False):
            reason = getattr(result, "error", "不明") if result is not None else "不明"
            self.send_text(int(chat_id), f"（返事を取れなかった: {str(reason)[:200]}）")
            return

        images = list(getattr(result, "images", []) or [])[: self.config.max_images]
        body = getattr(result, "text", "") or ""

        if images:
            # 画像と本文をまとめて送る。長い時は本文を別メッセージへ。
            caption = body if len(body) <= 1000 else ""
            self.send_photo(int(chat_id), images[0], caption=caption)
            if not caption and body:
                self.send_text(int(chat_id), body)
        elif body:
            self.send_text(int(chat_id), body)

        self.log(f"[tg] 返信 text={len(body)}字 images={len(images)}枚")

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.log(f"[tg] 待機開始 (allowed={self.config.allowed_chat_ids or '最初の相手を採用'})")
        while not self._stop:
            res = self._api(
                "getUpdates",
                {"offset": self._offset, "timeout": self.config.poll_timeout_sec},
                timeout=self.config.poll_timeout_sec + 15,
            )
            if not res.get("ok"):
                time.sleep(3)
                continue
            for upd in res.get("result", []):
                self._offset = max(self._offset, int(upd.get("update_id", 0)) + 1)
                msg = upd.get("message") or upd.get("edited_message")
                if msg:
                    try:
                        self._handle(msg)
                    except Exception as exc:
                        self.log(f"[tg] 処理で例外: {exc}")

    def stop(self) -> None:
        self._stop = True
