from __future__ import annotations

import unittest
from pathlib import Path


class PortableDefaultsTests(unittest.TestCase):
    def test_tracked_application_text_has_no_machine_specific_defaults(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = (
            "J:" + chr(92),
            "F:" + chr(92) + "kks",
            "F:/kks",
            "192.168.11.6",
            "192.168.11.10",
            "192.168.11.30",
        )
        suffixes = {".py", ".json", ".md", ".bat", ".txt"}
        application_roots = {
            "app", "config", "controllers", "core", "discord_bridge", "entrypoints",
            "grok_bridge", "grok_history", "gui", "orchestrator", "sd_prompt",
            "sd_prompt_receiver", "services", "telegram_bridge", "voice_gate_recorder",
            "workers",
        }
        root_files = {
            "README.md", "DISCORD_NEW_ACCOUNT.md", "config.sample.json",
            "discord_bridge_config.sample.json", "sd_prompt_receiver_config.sample.json",
            "run_telegram_bridge.py", "setup.bat", "setup_log.txt",
        }
        offenders: list[str] = []

        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            relative = path.relative_to(root)
            if not relative.parts:
                continue
            if relative.parts[0] not in application_roots and str(relative) not in root_files:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{relative}: {token}")

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
