"""PCM を「喋っている区間」に切り出す。

Discord から届くのは 48kHz / 2ch / 16bit の連続した PCM で、無音でも流れ続ける。
そのまま文字起こしへ渡すと無音まみれになるので、音量で区切る。

Discord に依存しない。PCM を食わせれば区間を返すだけなので、単体で試せる。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Utterance:
    """切り出した 1 区間。"""

    pcm: np.ndarray          # int16 モノラル。サンプリングレートは source_rate のまま
    source_rate: int
    started_at: float        # time.time()
    duration_sec: float
    user_id: int


class VadSegmenter:
    """
    音量(RMS)と無音長で区間を切る。

    しきい値を超えたら区間を開き、無音が silence_close_sec 続いたら閉じる。
    閉じた区間が min_utterance_sec に満たなければ雑音として捨てる。
    max_utterance_sec を超えたら、閉じないまま強制的に切り出す。

    pre_roll_sec ぶんは常に手前を保持しておき、区間の頭に足す。
    しきい値を超えた瞬間より前にある語頭の子音が切れるのを防ぐ。
    """

    def __init__(
        self,
        *,
        source_rate: int,
        rms_threshold: int,
        silence_close_sec: float,
        min_utterance_sec: float,
        max_utterance_sec: float,
        pre_roll_sec: float,
        on_utterance: Callable[[Utterance], None],
    ) -> None:
        self.source_rate = int(source_rate)
        self.rms_threshold = int(rms_threshold)
        self.silence_close_sec = float(silence_close_sec)
        self.min_utterance_sec = float(min_utterance_sec)
        self.max_utterance_sec = float(max_utterance_sec)
        self.pre_roll_samples = max(0, int(pre_roll_sec * self.source_rate))
        self.on_utterance = on_utterance

        self._open = False
        self._buf: list[np.ndarray] = []
        self._pre: list[np.ndarray] = []
        self._pre_len = 0
        self._silence_samples = 0
        self._started_at = 0.0
        self._user_id = 0
        # 実際に声が出ていたサンプル数。前後の詰め物を含めない。
        # これを長さの判定に使わないと、pre_roll と閉じ待ちの無音で
        # 短い雑音まで min_utterance_sec を通ってしまう。
        self._voiced_samples = 0

    # ------------------------------------------------------------------
    def feed(self, pcm_mono_int16: np.ndarray, user_id: int) -> None:
        """モノラル int16 を流し込む。長さは任意。"""
        if pcm_mono_int16.size == 0:
            return

        # 話者が変わったら、開いている区間を先に閉じる。
        if self._open and user_id != self._user_id:
            self._close(force=True)

        loud = self._rms(pcm_mono_int16) >= self.rms_threshold

        if not self._open:
            if loud:
                self._open = True
                self._user_id = user_id
                self._started_at = time.time()
                self._silence_samples = 0
                # 手前の保持分を頭に足す
                self._buf = list(self._pre)
                self._buf.append(pcm_mono_int16.copy())
                self._voiced_samples = pcm_mono_int16.size
                self._pre.clear()
                self._pre_len = 0
            else:
                self._keep_pre_roll(pcm_mono_int16)
            return

        # 区間が開いている
        self._buf.append(pcm_mono_int16.copy())

        if loud:
            self._silence_samples = 0
            self._voiced_samples += pcm_mono_int16.size
        else:
            self._silence_samples += pcm_mono_int16.size
            if self._silence_samples >= self.silence_close_sec * self.source_rate:
                self._close(force=False)
                return

        if self._buffered_sec() >= self.max_utterance_sec:
            self._close(force=True)

    def flush(self) -> None:
        """接続を切る時など、溜まっている分を吐き出す。"""
        if self._open:
            self._close(force=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        # int16 のまま二乗すると溢れるので float へ
        f = x.astype(np.float32)
        return float(np.sqrt(np.mean(f * f)))

    def _keep_pre_roll(self, chunk: np.ndarray) -> None:
        if self.pre_roll_samples <= 0:
            return
        self._pre.append(chunk.copy())
        self._pre_len += chunk.size
        while self._pre_len > self.pre_roll_samples and len(self._pre) > 1:
            dropped = self._pre.pop(0)
            self._pre_len -= dropped.size

    def _buffered_sec(self) -> float:
        n = sum(c.size for c in self._buf)
        return n / float(self.source_rate) if self.source_rate else 0.0

    def _close(self, *, force: bool) -> None:
        if not self._open:
            return
        self._open = False

        pcm = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.int16)
        self._buf = []
        trailing_silence = self._silence_samples
        voiced = self._voiced_samples
        self._silence_samples = 0
        self._voiced_samples = 0

        # 声が出ていた長さで判定する。詰め物込みの長さで測ると雑音が通る。
        voiced_sec = voiced / float(self.source_rate) if self.source_rate else 0.0
        if voiced_sec < self.min_utterance_sec:
            return

        # 閉じ待ちの無音を落とす。語尾が切れないよう少しだけ残す。
        keep_tail = int(0.2 * self.source_rate)
        drop = max(0, trailing_silence - keep_tail)
        if drop > 0 and pcm.size > drop:
            pcm = pcm[:-drop]

        dur = pcm.size / float(self.source_rate) if self.source_rate else 0.0

        self.on_utterance(
            Utterance(
                pcm=pcm,
                source_rate=self.source_rate,
                started_at=self._started_at,
                duration_sec=dur,
                user_id=self._user_id,
            )
        )
