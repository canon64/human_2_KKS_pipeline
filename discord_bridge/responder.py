"""返信の組み立て。

Human_2_kks から受け取った「テキスト + SD画像 + 音声ファイル」を、
テキストチャンネルと通話へ振り分ける。

Discord のオブジェクトには触れず、送信は呼び出し側から渡された
送信関数に任せる。こうしておくと discord ライブラリ無しで組み立てを試せる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def split_message(text: str, limit: int = 1900) -> list[str]:
    """Discord の文字数上限で割る。改行を優先し、無ければ強制的に切る。"""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]

    chunks: list[str] = []
    rest = normalized
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut <= 0:
            cut = rest.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
    return [c for c in chunks if c]


@dataclass
class ReplyPayload:
    """Human_2_kks から受け取る返信一式。"""

    text: str = ""
    # SD画像。バイト列で受ける（pipeline の extract_sd_result_images がこの形）。
    images: list[bytes] = field(default_factory=list)
    # 通話へ流す音声。TTS の出力ファイル。
    audio_path: str = ""
    # 付随情報（プロンプトなど）。ログ用。
    meta: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.text and not self.images and not self.audio_path


@dataclass
class ReplyPlan:
    """実際に何を送るかを決めた結果。送信の実行は呼び出し側が行う。"""

    text_chunks: list[str] = field(default_factory=list)
    image_files: list[tuple[str, bytes]] = field(default_factory=list)  # (filename, bytes)
    audio_path: str = ""
    skipped: list[str] = field(default_factory=list)


class Responder:
    def __init__(
        self,
        *,
        message_limit: int = 1900,
        max_images: int = 4,
        play_voice_in_call: bool = True,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.message_limit = max(300, min(2000, int(message_limit)))
        self.max_images = max(0, int(max_images))
        self.play_voice_in_call = bool(play_voice_in_call)
        self.log = log or (lambda _m: None)

    def build(self, payload: ReplyPayload) -> ReplyPlan:
        plan = ReplyPlan()

        if payload.text:
            plan.text_chunks = split_message(payload.text, self.message_limit)

        if payload.images:
            for i, blob in enumerate(payload.images, start=1):
                if len(plan.image_files) >= self.max_images:
                    plan.skipped.append(f"画像{i}枚目以降は上限({self.max_images})で送らない")
                    break
                if not blob:
                    continue
                plan.image_files.append((f"sd_{i:02d}.png", blob))

        if payload.audio_path:
            if not self.play_voice_in_call:
                plan.skipped.append("音声は設定で通話へ流さない")
            elif not Path(payload.audio_path).exists():
                plan.skipped.append(f"音声ファイルが無い: {payload.audio_path}")
            else:
                plan.audio_path = payload.audio_path

        self.log(
            f"[reply] text={len(plan.text_chunks)}分割 images={len(plan.image_files)}枚 "
            f"audio={'あり' if plan.audio_path else 'なし'}"
            + (f" skipped={plan.skipped}" if plan.skipped else "")
        )
        return plan
