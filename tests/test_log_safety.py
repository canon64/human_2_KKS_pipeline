import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.log_safety import (
    sanitize_log_text,
    summarize_sd_prompt_result,
    summarize_subprocess_error,
)


class LogSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image_text = "iVBORw0KGgo" + ("A" * 20_000)
        self.body = json.dumps({"images": [self.image_text]}, separators=(",", ":"))

    def test_sd_result_summary_omits_response_body(self) -> None:
        result = summarize_sd_prompt_result(
            {
                "ok": True,
                "status": 200,
                "url": "http://127.0.0.1:7860/sdapi/v1/txt2img",
                "body": self.body,
                "error": "",
            }
        )

        self.assertNotIn("body", result)
        self.assertTrue(result["body_omitted"])
        self.assertEqual(result["body_chars"], len(self.body))
        self.assertNotIn("iVBORw0KGgo", json.dumps(result))

    def test_subprocess_error_is_compact_and_excludes_base64(self) -> None:
        stdout = json.dumps(
            {
                "ok": False,
                "error": "HTTP Error 422: Unprocessable Entity",
                "event_stderr": "",
                "sd_prompt_send_result": {
                    "ok": True,
                    "status": 200,
                    "url": "http://127.0.0.1:7860/sdapi/v1/txt2img",
                    "body": self.body,
                    "error": "",
                },
            },
            ensure_ascii=False,
        )

        message = summarize_subprocess_error(stdout, "")

        self.assertIn("HTTP Error 422", message)
        self.assertIn("sd_status=200", message)
        self.assertIn(f"body_omitted={len(self.body)}chars", message)
        self.assertNotIn("iVBORw0KGgo", message)
        self.assertLess(len(message), 500)

    def test_gui_log_sanitizer_removes_json_body_and_standalone_base64(self) -> None:
        raw_json_log = '[error] payload={"body":' + json.dumps(self.body) + "}"
        safe_json_log = sanitize_log_text(raw_json_log)
        safe_base64_log = sanitize_log_text(f"image={self.image_text}")

        self.assertIn("<omitted body chars=", safe_json_log)
        self.assertNotIn("iVBORw0KGgo", safe_json_log)
        self.assertIn("<omitted base64 chars=", safe_base64_log)
        self.assertNotIn("iVBORw0KGgo", safe_base64_log)


if __name__ == "__main__":
    unittest.main()
