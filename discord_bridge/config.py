"""設定。

Human_2_kks から独立して動かせるよう、このライブラリ自身が全ての設定を持つ。
接続時は BridgeConfig を組み立てて渡すだけでよい。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class VoiceCaptureConfig:
    """音声受信と区間切り出し。"""

    # 受け取るボイスチャンネル。0 なら join しない。
    voice_channel_id: int = 0

    # このユーザーIDだけ拾う。空なら全員。
    # 通話に複数人いる時、自分の声だけを文字起こしへ回したい場合に使う。
    target_user_ids: list[int] = field(default_factory=list)

    # 区間の切り出し。Discord から届く PCM は 48kHz / 2ch / 16bit。
    # RMS がこの値を超えている間を「喋っている」とみなす。0-32767 の尺度。
    rms_threshold: int = 900

    # 無音がこの秒数続いたら区間を閉じる。短いと語尾が切れ、長いと反応が遅れる。
    silence_close_sec: float = 0.6

    # 短すぎる区間は雑音として捨てる。
    min_utterance_sec: float = 0.4

    # 長すぎる区間は途中で強制的に閉じる（延々と閉じない事故を防ぐ）。
    max_utterance_sec: float = 30.0

    # 区間の頭を少し遡って含める。しきい値を超えた瞬間より前の子音が切れるのを防ぐ。
    pre_roll_sec: float = 0.25


@dataclass
class WavOutputConfig:
    """WAV の書き出し先と形式。"""

    # 書き出し先。Human_2_kks の監視フォルダを指すと、そのまま文字起こしへ流れる。
    # パイプラインは watchdog でこのフォルダを見ているので、こちらは置くだけでよい。
    output_dir: str = ""

    # faster-whisper が扱いやすい形へ落とす。
    sample_rate: int = 16000
    channels: int = 1

    # ファイル名の頭。どこから来た音声かをログで見分けるため。
    filename_prefix: str = "discord"

    # 書き出しを一時ファイルで行い、完了後に改名する。
    # watchdog は on_created で拾うので、書き込み途中のファイルを掴ませない。
    use_temp_then_rename: bool = True


@dataclass
class ReplyConfig:
    """返信の送り先。"""

    # テキストと画像を送るチャンネル。0 なら送らない。
    text_channel_id: int = 0

    # Discord の 1 メッセージあたりの上限。実際は 2000 だが余裕を持たせる。
    message_limit: int = 1900

    # 画像の添付枚数の上限。
    max_images: int = 4

    # 音声を通話へ流すか。False ならテキストのみ。
    play_voice_in_call: bool = True


@dataclass
class PipelineLink:
    """Human_2_kks への接続先。/ask に投げて返事を待つ。"""

    host: str = "127.0.0.1"
    port: int = 18767
    token: str = ""
    timeout_sec: float = 180.0

    # このチャンネルのメッセージを拾う。0 なら reply.text_channel_id と同じ。
    listen_channel_id: int = 0

    # 頭にこれが付いたメッセージだけ拾う。空なら全部。
    command_prefix: str = ""

    # Bot 自身と他の Bot の発言は無視する。
    ignore_bots: bool = True

    # 動作を許可するサーバーID。空なら制限しない。
    # 他のエージェント用サーバーへ誤って流さないための歯止め。
    # ここに入れたサーバー以外では、受信も送信も一切しない。
    allowed_guild_ids: list[int] = field(default_factory=list)

    # 音声メッセージの文字起こし先。パイプラインが使う RTFW LAN と同じ。
    rtfw_host: str = "192.168.11.30"
    rtfw_port: int = 8766

    # 改行の置き換え文字。
    # Grok の入力欄は Enter が送信操作なので、改行を含んだまま打ち込むと
    # 途中で確定して入力が壊れ、送信ボタンが無効のまま残る。
    # Discord は複数行が普通に書けるため、入口で1行へ潰す。
    newline_replacement: str = " "


@dataclass
class BotProfileConfig:
    """Discord Bot の表示プロフィール。"""

    username: str = ""
    avatar_path: str = ""


@dataclass
class BridgeConfig:
    # Bot トークンを読む環境変数名。トークン自体はここに置かない。
    token_env: str = "DISCORD_BOT_TOKEN"

    # .env の場所。abc_canvas と同じ置き場を既定にする。
    env_file: str = r"J:\tools\api-scripts\runtime\.env"

    capture: VoiceCaptureConfig = field(default_factory=VoiceCaptureConfig)
    wav: WavOutputConfig = field(default_factory=WavOutputConfig)
    reply: ReplyConfig = field(default_factory=ReplyConfig)
    pipeline: PipelineLink = field(default_factory=PipelineLink)
    profile: BotProfileConfig = field(default_factory=BotProfileConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str | Path) -> "BridgeConfig":
        p = Path(path)
        if not p.exists():
            return BridgeConfig()
        raw = json.loads(p.read_text(encoding="utf-8"))
        return BridgeConfig(
            token_env=raw.get("token_env", "DISCORD_BOT_TOKEN"),
            env_file=raw.get("env_file", ""),
            capture=VoiceCaptureConfig(**(raw.get("capture") or {})),
            wav=WavOutputConfig(**(raw.get("wav") or {})),
            reply=ReplyConfig(**(raw.get("reply") or {})),
            pipeline=PipelineLink(**(raw.get("pipeline") or {})),
            profile=BotProfileConfig(**(raw.get("profile") or {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")
