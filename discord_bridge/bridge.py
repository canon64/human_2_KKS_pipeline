"""Discord との接続部。

ここだけが discord ライブラリに依存する。import は遅延させてあるので、
未インストールでも他のモジュール（segmenter / wav_writer / responder）は動く。

音声の受信は標準の discord.py では出来ない。py-cord の discord.sinks を使う。
    pip install py-cord[voice]
py-cord と discord.py は同じ `discord` 名前空間を使うため同居できない。

Human_2_kks との接続点は2つだけ:
    on_wav_ready(path)      … 区間を WAV にした直後。監視フォルダへ置くなら何もしなくてよい
    request_reply(text)     … テキストを渡して ReplyPayload を返してもらう
どちらも呼び出し側が差し込む。このライブラリは Human_2_kks を import しない。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import BridgeConfig
from .pipeline_client import PipelineClient, PipelineConfig
from .responder import ReplyPayload, ReplyPlan, Responder
from .segmenter import Utterance, VadSegmenter
from .wav_writer import WavWriter

# Discord のボイスは 48kHz / 2ch / 16bit で届く
DISCORD_PCM_RATE = 48000
DISCORD_PCM_CHANNELS = 2


def load_env_file(path: str) -> int:
    """.env を読んで os.environ へ入れる。既存の値は上書きしない。"""
    p = Path(path)
    if not path or not p.exists():
        return 0
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


def stereo_to_mono(pcm_bytes: bytes) -> np.ndarray:
    """48kHz/2ch/16bit のバイト列を int16 モノラルへ。"""
    if not pcm_bytes:
        return np.zeros(0, dtype=np.int16)
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    if arr.size == 0:
        return arr
    if DISCORD_PCM_CHANNELS > 1:
        usable = (arr.size // DISCORD_PCM_CHANNELS) * DISCORD_PCM_CHANNELS
        if usable == 0:
            return np.zeros(0, dtype=np.int16)
        arr = arr[:usable].reshape(-1, DISCORD_PCM_CHANNELS)
        # 平均を取る。int16 のまま足すと溢れるので int32 を経由
        arr = arr.astype(np.int32).mean(axis=1)
        arr = np.clip(arr, -32768, 32767).astype(np.int16)
    return arr


class VoiceBridge:
    """
    ボイスチャンネルの音声を区間ごとに WAV へ落とし、
    Human_2_kks から返ってきたテキスト/画像/音声を送り返す。

    使い方:
        bridge = VoiceBridge(config)
        bridge.on_wav_ready = lambda path: ...      # 省略可
        bridge.request_reply = lambda text: payload # 省略可
        bridge.start()
    """

    def __init__(
        self,
        config: BridgeConfig,
        *,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.log = log or (lambda m: print(m))

        self.writer = WavWriter(
            output_dir=config.wav.output_dir,
            sample_rate=config.wav.sample_rate,
            channels=config.wav.channels,
            filename_prefix=config.wav.filename_prefix,
            use_temp_then_rename=config.wav.use_temp_then_rename,
        )
        self.responder = Responder(
            message_limit=config.reply.message_limit,
            max_images=config.reply.max_images,
            play_voice_in_call=config.reply.play_voice_in_call,
            log=self.log,
        )

        self.pipeline = PipelineClient(
            PipelineConfig(
                host=config.pipeline.host,
                port=config.pipeline.port,
                token=config.pipeline.token,
                timeout_sec=config.pipeline.timeout_sec,
            ),
            log=self.log,
        )

        # 接続点。差し込まれなければ何もしない。
        self.on_wav_ready: Callable[[Path], None] | None = None
        self.request_reply: Callable[[str], ReplyPayload | None] | None = None

        # 同じメッセージを二重に処理しないための記録。
        # discord.Bot は commands.Bot 派生で、@client.event の登録と内部処理の
        # 両方が走ることがあり、同じ ID が2回来る。
        self._seen_message_ids: list[int] = []

        self._segmenters: dict[int, VadSegmenter] = {}
        self._client: Any | None = None
        self._thread: threading.Thread | None = None
        self._voice_client: Any | None = None

    # ------------------------------------------------------------------
    # 音声の入口。Discord の sink から呼ぶ。テスト時は直接呼んでよい。
    def feed_pcm(self, pcm_bytes: bytes, user_id: int) -> None:
        cap = self.config.capture
        if cap.target_user_ids and user_id not in cap.target_user_ids:
            return

        mono = stereo_to_mono(pcm_bytes)
        if mono.size == 0:
            return

        seg = self._segmenters.get(user_id)
        if seg is None:
            seg = VadSegmenter(
                source_rate=DISCORD_PCM_RATE,
                rms_threshold=cap.rms_threshold,
                silence_close_sec=cap.silence_close_sec,
                min_utterance_sec=cap.min_utterance_sec,
                max_utterance_sec=cap.max_utterance_sec,
                pre_roll_sec=cap.pre_roll_sec,
                on_utterance=self._handle_utterance,
            )
            self._segmenters[user_id] = seg

        seg.feed(mono, user_id)

    def flush_all(self) -> None:
        for seg in self._segmenters.values():
            seg.flush()

    def _handle_utterance(self, utt: Utterance) -> None:
        path = self.writer.write(utt)
        if path is None:
            self.log(f"[voice] 書き出し先が未設定 ({utt.duration_sec:.2f}s を捨てた)")
            return

        self.log(f"[voice] {utt.duration_sec:.2f}s user={utt.user_id} -> {path.name}")

        if self.on_wav_ready is not None:
            try:
                self.on_wav_ready(path)
            except Exception as exc:
                self.log(f"[voice] on_wav_ready で例外: {exc}")

    # ------------------------------------------------------------------
    # 返信。Human_2_kks から結果を貰って送る。
    def send_reply(self, payload: ReplyPayload) -> ReplyPlan:
        plan = self.responder.build(payload)
        if self._client is None:
            self.log("[reply] Discord 未接続なので組み立てのみ")
            return plan
        self._dispatch(plan)
        return plan

    def _dispatch(self, plan: ReplyPlan) -> None:
        """組み立て済みの内容を実際に送る。discord のイベントループへ渡す。"""
        import asyncio

        client = self._client
        if client is None:
            return

        async def _run() -> None:
            ch_id = self.config.reply.text_channel_id
            if ch_id and (plan.text_chunks or plan.image_files):
                channel = await self._resolve_channel(client, ch_id)
                if channel is None:
                    self.log(f"[reply] テキストチャンネルが取れない: {ch_id}")
                else:
                    import discord  # 遅延 import

                    for chunk in plan.text_chunks:
                        await channel.send(chunk)
                    if plan.image_files:
                        import io as _io

                        files = [
                            discord.File(_io.BytesIO(blob), filename=name)
                            for name, blob in plan.image_files
                        ]
                        await channel.send(files=files)

            if plan.audio_path and self._voice_client is not None:
                import discord  # 遅延 import

                try:
                    source = discord.FFmpegPCMAudio(plan.audio_path)
                    if self._voice_client.is_playing():
                        self._voice_client.stop()
                    self._voice_client.play(source)
                except Exception as exc:
                    self.log(f"[reply] 通話への再生に失敗: {exc}")

        asyncio.run_coroutine_threadsafe(_run(), client.loop)

    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Bot を別スレッドで起動する。discord が無ければ False。"""
        try:
            import discord  # noqa: F401
        except ImportError:
            self.log(
                "[bridge] discord が入っていない。音声受信には py-cord が要る:\n"
                "         pip install py-cord[voice]"
            )
            return False

        load_env_file(self.config.env_file)
        token = os.environ.get(self.config.token_env, "").strip()
        if not token:
            self.log(f"[bridge] トークンが無い（環境変数 {self.config.token_env}）")
            return False

        self._thread = threading.Thread(target=self._run_client, args=(token,), daemon=True)
        self._thread.start()
        return True

    @staticmethod
    def _ensure_opus(log: Callable[[str], None]) -> bool:
        """音声の送受信には opus が要る。Windows では同梱版を明示的に読む。"""
        import discord

        if discord.opus.is_loaded():
            return True
        try:
            discord.opus._load_default()
        except Exception:
            pass
        if discord.opus.is_loaded():
            log("[bridge] opus を読み込んだ")
            return True
        log("[bridge] opus が読み込めない。音声の送受信ができない")
        return False

    def _run_client(self, token: str) -> None:
        import asyncio

        import discord

        # 別スレッドにはイベントループが無い。クライアントを作る前に用意して紐付ける。
        # discord.Bot() は生成時に get_event_loop() を呼ぶので、後からでは間に合わない。
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self._ensure_opus(self.log)

        intents = discord.Intents.default()
        intents.message_content = True   # メッセージ本文を読むのに必須
        intents.voice_states = True
        client = discord.Bot(intents=intents) if hasattr(discord, "Bot") else discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_message(message) -> None:  # type: ignore[misc, no-untyped-def]
            await self._handle_message(client, message)

        @client.event
        async def on_ready() -> None:  # type: ignore[misc]
            self.log(f"[bridge] 接続した: {client.user}")
            ch_id = self.config.capture.voice_channel_id
            if not ch_id:
                return
            channel = client.get_channel(ch_id)
            if channel is None:
                self.log(f"[bridge] ボイスチャンネルが見つからない: {ch_id}")
                return
            try:
                self._voice_client = await channel.connect()
                self._start_recording()
            except Exception as exc:
                self.log(f"[bridge] 通話へ入れない: {exc}")

        try:
            loop.run_until_complete(client.start(token))
        except Exception as exc:
            self.log(f"[bridge] クライアントが停止した: {exc}")
        finally:
            try:
                loop.run_until_complete(client.close())
            except Exception:
                pass
            loop.close()

    def list_channels(self, timeout: float = 30.0) -> list[dict[str, Any]]:
        """
        Bot が見える全チャンネルを列挙して返す。設定に入れる ID を調べるため。
        接続して取得したら切断する。
        """
        try:
            import discord
        except ImportError:
            self.log("[bridge] discord が入っていない")
            return []

        load_env_file(self.config.env_file)
        token = os.environ.get(self.config.token_env, "").strip()
        if not token:
            self.log(f"[bridge] トークンが無い（環境変数 {self.config.token_env}）")
            return []

        import asyncio

        found: list[dict[str, Any]] = []

        async def _main() -> None:
            intents = discord.Intents.default()
            intents.guilds = True
            client = discord.Client(intents=intents)

            @client.event
            async def on_ready() -> None:  # type: ignore[misc]
                for guild in client.guilds:
                    for ch in guild.channels:
                        found.append(
                            {
                                "guild": guild.name,
                                "guild_id": guild.id,
                                "name": ch.name,
                                "id": ch.id,
                                "type": type(ch).__name__,
                            }
                        )
                await client.close()

            try:
                await asyncio.wait_for(client.start(token), timeout=timeout)
            except asyncio.TimeoutError:
                self.log("[bridge] 一覧取得がタイムアウトした")
            except Exception as exc:
                # 取得後に自分で close() した直後の "Session is closed" は正常終了。
                if "Session is closed" not in str(exc):
                    self.log(f"[bridge] 一覧取得に失敗: {exc}")
            finally:
                if not client.is_closed():
                    await client.close()

        asyncio.run(_main())
        return found

    async def _resolve_channel(self, client: Any, channel_id: int) -> Any:
        """
        チャンネルを取る。get_channel はキャッシュにしか当たらず、
        Bot が起動直後などでは None を返す。その場合は API から取りに行く。
        権限が無い場合はここで Forbidden になるので、理由がログに出る。
        """
        channel = client.get_channel(channel_id)
        if channel is not None:
            return channel
        try:
            return await client.fetch_channel(channel_id)
        except Exception as exc:
            name = type(exc).__name__
            if name == "Forbidden":
                self.log(f"[reply] チャンネルの閲覧権限が無い: {channel_id}")
            else:
                self.log(f"[reply] チャンネル取得に失敗 ({name}): {channel_id}")
            return None

    async def _handle_audio_attachment(self, message: Any) -> bool:
        """
        音声の添付があれば、文字起こしして /ask へ渡す。
        扱ったら True を返す。テキストとして続行する場合は False。

        文字起こしは RTFW LAN (パイプラインが使っている LAN 上のサーバー) に任せる。
        監視フォルダへ置く方式だとゲーム内の字幕/TTSへ流れて Discord に返らないので、
        ここで先にテキストへ変換してから通常の経路へ乗せる。
        """
        import asyncio
        import tempfile
        import uuid
        from pathlib import Path

        atts = list(getattr(message, "attachments", None) or [])
        target = None
        for a in atts:
            ct = (getattr(a, "content_type", "") or "").lower()
            name = (getattr(a, "filename", "") or "").lower()
            if ct.startswith("audio") or name.endswith((".ogg", ".oga", ".mp3", ".m4a", ".wav", ".webm")):
                target = a
                break
        if target is None:
            return False

        self.log(f"[voice-msg] 受信 {getattr(target, 'filename', '?')}")
        try:
            await message.channel.send("（音声を受け取った。文字起こし中…）")
        except Exception:
            pass

        tmp = Path(tempfile.gettempdir()) / f"dc_{uuid.uuid4().hex}_{getattr(target,'filename','a.ogg')}"
        try:
            await target.save(tmp)
        except Exception as exc:
            self.log(f"[voice-msg] 保存に失敗: {exc}")
            await message.channel.send("（音声を取得できなかった）")
            return True

        wav = await asyncio.to_thread(self._convert_to_wav, tmp)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        if wav is None:
            await message.channel.send("（音声を変換できなかった）")
            return True

        text = await asyncio.to_thread(self._transcribe, wav)
        if not text:
            await message.channel.send("（文字起こしできなかった）")
            return True

        self.log(f"[voice-msg] 文字起こし: {text[:60]}")
        try:
            await message.channel.send(f"（聞き取り: {text[:300]}）")
        except Exception:
            pass

        result = await asyncio.to_thread(self.pipeline.ask, text)
        if result is None or not result.ok:
            reason = result.error if result is not None else "不明"
            await message.channel.send(f"（返事を取れなかった: {str(reason)[:200]}）")
            return True

        plan = self.responder.build(
            ReplyPayload(text=result.text, images=result.images,
                         audio_path=result.audio_path,
                         meta={"sd_prompt": result.sd_prompt})
        )
        await self._send_plan(message.channel, plan)
        return True

    def _convert_to_wav(self, src: "Path") -> "Path | None":
        """faster-whisper が扱いやすい 16kHz モノラルへ落とす。"""
        import subprocess
        import tempfile
        import uuid
        from pathlib import Path

        dst = Path(tempfile.gettempdir()) / f"dc_{uuid.uuid4().hex}.wav"
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
                capture_output=True, timeout=120,
            )
            if proc.returncode != 0 or not dst.exists():
                self.log(f"[voice-msg] ffmpeg 失敗: {proc.stderr.decode(errors='replace')[:160]}")
                return None
            return dst
        except Exception as exc:
            self.log(f"[voice-msg] ffmpeg 実行に失敗: {exc}")
            return None

    def _transcribe(self, wav: "Path") -> str:
        """RTFW LAN で文字起こしする。パイプラインが使っているのと同じ経路。"""
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from services.rtfw_lan_service import transcribe_wav_rtfw
        except Exception as exc:
            self.log(f"[voice-msg] 文字起こしの読み込みに失敗: {exc}")
            return ""

        link = self.config.pipeline
        host = getattr(link, "rtfw_host", "") or "192.168.11.30"
        port = int(getattr(link, "rtfw_port", 0) or 8766)
        try:
            res = transcribe_wav_rtfw(wav, host=host, port=port)
        except Exception as exc:
            self.log(f"[voice-msg] 文字起こしで例外: {exc}")
            return ""
        finally:
            try:
                wav.unlink(missing_ok=True)
            except Exception:
                pass

        if not res.get("ok"):
            self.log(f"[voice-msg] 文字起こし失敗: {str(res.get('error'))[:160]}")
            return ""
        return str(res.get("text", "") or "").strip()

    async def _handle_message(self, client: Any, message: Any) -> None:
        """テキストを受けて Human_2_kks へ投げ、返事と画像を返す。"""
        import asyncio

        link = self.config.pipeline
        if link.ignore_bots and getattr(message.author, "bot", False):
            return
        if client.user is not None and message.author.id == client.user.id:
            return

        # 許可したサーバー以外では動かない。設定を間違えても他所へ流れない。
        allowed = getattr(link, "allowed_guild_ids", None) or []
        if allowed:
            gid = getattr(getattr(message, "guild", None), "id", 0)
            if gid not in allowed:
                return

        listen_id = link.listen_channel_id or self.config.reply.text_channel_id
        if listen_id and message.channel.id != listen_id:
            return

        mid = getattr(message, "id", 0)
        if mid:
            if mid in self._seen_message_ids:
                return  # すでに処理済み
            self._seen_message_ids.append(mid)
            if len(self._seen_message_ids) > 200:
                del self._seen_message_ids[:100]

        # 音声の添付(ボイスメッセージ/音声ファイル)を先に処理する。
        # 通話の受信は暗号化で塞がれているが、添付は普通に取れる。
        if await self._handle_audio_attachment(message):
            return

        text = (message.content or "").strip()
        if link.command_prefix:
            if not text.startswith(link.command_prefix):
                return
            text = text[len(link.command_prefix):].strip()
        if not text:
            return

        # Grok の入力欄は Enter が送信になるため、改行を含むと入力が壊れる。
        # 1行へ潰してから渡す。元の見た目は Discord 側に残る。
        raw_lines = text.count(chr(10)) + 1
        if chr(10) in text or chr(13) in text:
            sep = link.newline_replacement if link.newline_replacement is not None else " "
            norm = text.replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))
            text = sep.join(p.strip() for p in norm.split(chr(10)) if p.strip())

        _note = (" (" + str(raw_lines) + "行を1行へ)") if raw_lines > 1 else ""
        self.log("[msg] 受信 " + str(message.author) + ": " + text[:60] + _note)

        # Grok と SD で数十秒かかる。待っていることが見えるようにする。
        typing_ctx = None
        try:
            typing_ctx = message.channel.typing()
            await typing_ctx.__aenter__()
        except Exception:
            typing_ctx = None

        try:
            # HTTP は同期なので、イベントループを塞がないよう別スレッドで待つ。
            result = await asyncio.to_thread(self.pipeline.ask, text)
        except Exception as exc:
            self.log(f"[msg] 問い合わせで例外: {exc}")
            result = None
        finally:
            if typing_ctx is not None:
                try:
                    await typing_ctx.__aexit__(None, None, None)
                except Exception:
                    pass

        if result is None or not result.ok:
            reason = result.error if result is not None else "不明"
            self.log(f"[msg] 失敗: {reason}")
            try:
                await message.channel.send(f"（返事を取れなかった: {reason[:150]}）")
            except Exception:
                pass
            return

        plan = self.responder.build(
            ReplyPayload(text=result.text, images=result.images,
                         audio_path=result.audio_path,
                         meta={"sd_prompt": result.sd_prompt})
        )
        await self._send_plan(message.channel, plan)

    async def _send_plan(self, channel: Any, plan: ReplyPlan) -> None:
        import io as _io

        import discord

        for chunk in plan.text_chunks:
            await channel.send(chunk)

        if plan.image_files:
            files = [
                discord.File(_io.BytesIO(blob), filename=name)
                for name, blob in plan.image_files
            ]
            await channel.send(files=files)

        # TTS の音声をファイルとして送る。通話ではないので暗号化の制約を受けない。
        if plan.audio_path:
            from pathlib import Path as _P

            p = _P(plan.audio_path)
            sent = False
            try:
                if p.exists():
                    await channel.send(file=discord.File(str(p), filename=p.name))
                    self.log(f"[reply] 音声を送った: {p.name}")
                    sent = True
                else:
                    self.log(f"[reply] 音声ファイルが無い: {p}")
            except Exception as exc:
                self.log(f"[reply] 音声の送信に失敗: {exc}")

            # 送信後に結合ファイルを消す。元の parts/line_*.wav は
            # ゲーム側の再生に使うので触らない。消すのは外部送信用に作った1本だけ。
            if sent and p.name == "joined_for_external.wav":
                try:
                    p.unlink(missing_ok=True)
                    self.log(f"[reply] 結合ファイルを消した: {p.name}")
                except Exception as exc:
                    self.log(f"[reply] 結合ファイルを消せない: {exc}")

    def _start_recording(self) -> None:
        """py-cord の sink で受信を開始する。"""
        import discord

        vc = self._voice_client
        if vc is None:
            return

        bridge = self

        class _Sink(discord.sinks.Sink):  # type: ignore[attr-defined]
            def write(self, data, user):  # noqa: D401
                try:
                    bridge.feed_pcm(bytes(data), int(user))
                except Exception as exc:
                    bridge.log(f"[voice] sink で例外: {exc}")

        async def _done(sink, *args):  # noqa: ANN001
            bridge.flush_all()

        try:
            vc.start_recording(_Sink(), _done)
            self.log("[bridge] 音声の受信を開始した")
        except AttributeError:
            self.log(
                "[bridge] 音声受信は使えない（Discord の E2E 暗号化 DAVE により"
                "py-cord 側が未対応）。テキストの送受信には影響しない"
            )
        except Exception as exc:
            msg = str(exc)
            if "DAVE" in msg or "reception" in msg.lower():
                self.log("[bridge] 音声受信は Discord の E2E 暗号化により現在不可。テキストは動く")
            else:
                self.log(f"[bridge] 受信開始に失敗: {exc}")
