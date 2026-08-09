"""区間を WAV にして書き出す。

Human_2_kks は watchdog の on_created でフォルダを監視している。
書き込み途中のファイルを掴まれると壊れた WAV を読ませることになるので、
一時名で書いてから改名する（改名は on_created ではなく on_moved になるが、
多くの監視は最終的なファイル出現を拾える。確実を期すため既定で有効）。

faster-whisper に合わせて 16kHz モノラルへ落とす。
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np

from .segmenter import Utterance


def resample_int16(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """線形補間で落とす。48k→16k のような整数比でなくても通る。"""
    if src_rate == dst_rate or pcm.size == 0:
        return pcm.astype(np.int16, copy=False)

    dst_len = int(round(pcm.size * (dst_rate / float(src_rate))))
    if dst_len <= 0:
        return np.zeros(0, dtype=np.int16)

    src_idx = np.linspace(0.0, pcm.size - 1.0, num=dst_len, dtype=np.float64)
    out = np.interp(src_idx, np.arange(pcm.size, dtype=np.float64), pcm.astype(np.float64))
    return np.clip(out, -32768, 32767).astype(np.int16)


class WavWriter:
    def __init__(
        self,
        *,
        output_dir: str,
        sample_rate: int = 16000,
        channels: int = 1,
        filename_prefix: str = "discord",
        use_temp_then_rename: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.sample_rate = int(sample_rate)
        self.channels = max(1, int(channels))
        self.filename_prefix = filename_prefix or "discord"
        self.use_temp_then_rename = bool(use_temp_then_rename)
        self._seq = 0

    def write(self, utt: Utterance) -> Path | None:
        """書き出したパスを返す。出力先が未設定なら None。"""
        if not str(self.output_dir):
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        pcm = resample_int16(utt.pcm, utt.source_rate, self.sample_rate)
        if pcm.size == 0:
            return None

        # 同じミリ秒に複数の区間が閉じると名前が衝突するので連番を足す。
        self._seq = (self._seq + 1) % 10000
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(utt.started_at))
        ms = int((utt.started_at % 1.0) * 1000)
        name = f"{self.filename_prefix}_{stamp}_{ms:03d}_{self._seq:04d}_u{utt.user_id}.wav"
        final_path = self.output_dir / name

        write_path = (
            self.output_dir / (name + ".part") if self.use_temp_then_rename else final_path
        )

        with wave.open(str(write_path), "wb") as w:
            w.setnchannels(self.channels)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            if self.channels == 1:
                w.writeframes(pcm.tobytes())
            else:
                stacked = np.repeat(pcm[:, None], self.channels, axis=1)
                w.writeframes(stacked.tobytes())

        if self.use_temp_then_rename:
            write_path.replace(final_path)

        return final_path
