"""Human_2_kks への問い合わせ。

パイプラインの /ask に投げて、返事とSD画像を受け取る。
/manual-text は投げっぱなしで結果が返らないので、返信が要る用途ではこちらを使う。

このモジュールは Human_2_kks を import しない。HTTP だけで話す。
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PipelineConfig:
    host: str = "127.0.0.1"
    port: int = 8767
    token: str = ""
    # Grok の応答とSD生成が終わるまで待つ。生成が重い時は伸ばす。
    timeout_sec: float = 180.0


@dataclass
class AskResult:
    ok: bool = False
    text: str = ""
    images: list[bytes] = field(default_factory=list)
    sd_prompt: str = ""
    audio_path: str = ""
    error: str = ""


class PipelineClient:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.log = log or (lambda _m: None)

    def ask(self, text: str) -> AskResult:
        url = f"http://{self.config.host}:{self.config.port}/ask"
        body = json.dumps(
            {"text": text, "timeout_sec": self.config.timeout_sec},
            ensure_ascii=False,
        ).encode("utf-8")

        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.config.token:
            req.add_header("X-Auth-Token", self.config.token)

        # HTTP 側は少し長めに待つ。パイプライン側のタイムアウトを先に効かせたい。
        http_timeout = self.config.timeout_sec + 15.0

        try:
            with urllib.request.urlopen(req, timeout=http_timeout) as res:
                raw = res.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:200]
            except Exception:
                pass
            self.log(f"[pipeline] HTTP {e.code}: {detail}")
            return AskResult(ok=False, error=f"HTTP {e.code} {detail}")
        except Exception as exc:
            self.log(f"[pipeline] 繋がらない: {exc}")
            return AskResult(ok=False, error=str(exc))

        try:
            data = json.loads(raw)
        except Exception:
            return AskResult(ok=False, error="応答が JSON でない")

        if not data.get("ok"):
            return AskResult(ok=False, error=str(data.get("error", "")))

        images: list[bytes] = []
        for b64 in data.get("images") or []:
            try:
                images.append(base64.b64decode(b64))
            except Exception:
                continue

        return AskResult(
            ok=True,
            text=str(data.get("text", "") or ""),
            images=images,
            sd_prompt=str(data.get("sd_prompt", "") or ""),
            audio_path=str(data.get("audio_path", "") or ""),
        )
