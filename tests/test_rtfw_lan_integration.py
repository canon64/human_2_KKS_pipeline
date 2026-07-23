from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.models import AppConfig
from services.rtfw_lan_service import dispatch_transcription, wav_to_pcm16_mono_16k
from voice_gate_recorder import RecorderConfig, VoiceGateRecorder


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.astype("<i2", copy=False).tobytes())


class RtfwClipIntegrationTest(unittest.TestCase):
    def test_only_existing_device_config_remains(self) -> None:
        fields = set(AppConfig.__dataclass_fields__)
        self.assertIn("device", fields)
        self.assertNotIn("rtfw_device_id", fields)
        self.assertNotIn("rtfw_source", fields)

    def test_wav_is_converted_to_protocol_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "clip.wav"
            _write_wav(wav_path, np.arange(3200, dtype=np.int16), sample_rate=16000)
            pcm = wav_to_pcm16_mono_16k(wav_path)
        self.assertEqual(6400, len(pcm))

    def test_local_and_remote_use_one_dispatch_function(self) -> None:
        wav_path = Path("same.wav")
        local_calls: list[Path] = []
        local_cfg = SimpleNamespace(fw_backend="local")
        remote_cfg = SimpleNamespace(
            fw_backend="rtfw_lan", rtfw_host="192.168.11.6", rtfw_port=8766
        )

        local_result = dispatch_transcription(
            local_cfg,
            wav_path,
            local_transcriber=lambda path: local_calls.append(path) or {"ok": True, "text": "local"},
        )
        with patch("services.rtfw_lan_service.transcribe_wav_rtfw") as remote:
            remote.return_value = {"ok": True, "text": "remote", "backend": "rtfw_lan"}
            remote_result = dispatch_transcription(
                remote_cfg,
                wav_path,
                local_transcriber=lambda _path: self.fail("local backend must not run"),
            )

        self.assertEqual([wav_path], local_calls)
        self.assertEqual("local", local_result["text"])
        self.assertEqual("remote", remote_result["text"])
        remote.assert_called_once()

    def test_vr_ptt_release_wav_enters_remote_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            recorder = VoiceGateRecorder(RecorderConfig(
                output_dir=output,
                sample_rate=16000,
                block_ms=100,
                threshold_dbfs=-27.5,
                silence_seconds=1.75,
                min_duration_seconds=0.1,
                pre_roll_seconds=0.65,
                device=None,
                external_control_enabled=True,
                external_control_strict_hold=True,
            ))
            self.assertEqual(-27.5, recorder.cfg.threshold_dbfs)
            self.assertEqual(1.75, recorder.cfg.silence_seconds)
            self.assertEqual(0.1, recorder.cfg.min_duration_seconds)
            self.assertEqual(0.65, recorder.cfg.pre_roll_seconds)
            recorder._control_q.put_nowait("start")
            recorder._drain_control_queue()
            for _ in range(3):
                recorder.process_chunk(np.full(1600, 1000, dtype=np.int16))
            recorder._control_q.put_nowait("stop")
            recorder._drain_control_queue()
            wavs = list(output.glob("*.wav"))
            self.assertEqual(1, len(wavs))

            cfg = SimpleNamespace(
                fw_backend="rtfw_lan", rtfw_host="192.168.11.6", rtfw_port=8766
            )
            with patch("services.rtfw_lan_service.transcribe_wav_rtfw") as remote:
                remote.return_value = {"ok": True, "text": "ptt final", "backend": "rtfw_lan"}
                result = dispatch_transcription(
                    cfg,
                    wavs[0],
                    local_transcriber=lambda _path: self.fail("local backend must not run"),
                )
            self.assertEqual("ptt final", result["text"])
            remote.assert_called_once()
            remote_kwargs = remote.call_args.kwargs
            self.assertNotIn("threshold_dbfs", remote_kwargs)
            self.assertNotIn("silence_seconds", remote_kwargs)
            self.assertNotIn("min_duration_seconds", remote_kwargs)
            self.assertNotIn("pre_roll_seconds", remote_kwargs)


if __name__ == "__main__":
    unittest.main()
