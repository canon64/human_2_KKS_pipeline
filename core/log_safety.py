from __future__ import annotations

import json
import re
from typing import Any


_JSON_BODY_FIELD_RE = re.compile(
    r'("body"\s*:\s*)"(?:\\.|[^"\\])*"',
    re.DOTALL,
)
_LONG_BASE64_RE = re.compile(
    r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{512,}={0,2}(?![A-Za-z0-9+/=])'
)


def summarize_sd_prompt_result(value: object) -> dict[str, Any]:
    """Return SD result metadata without embedding image response bodies."""
    if not isinstance(value, dict):
        return {}

    summary = dict(value)
    body = summary.pop("body", "")
    if body not in (None, ""):
        body_text = str(body)
        summary["body_omitted"] = True
        summary["body_chars"] = len(body_text)

    images = summary.pop("images", None)
    if isinstance(images, list):
        summary["images_omitted"] = len(images)
    elif images not in (None, ""):
        summary["images_omitted"] = True
    return summary


def sanitize_log_text(value: object, *, max_chars: int = 16_000) -> str:
    """Remove image payloads and bound a message before it reaches GUI/file logs."""
    text = str(value or "")

    def omit_body(match: re.Match[str]) -> str:
        encoded_chars = max(0, len(match.group(0)) - len(match.group(1)) - 2)
        return f'{match.group(1)}"<omitted body chars={encoded_chars}>"'

    text = _JSON_BODY_FIELD_RE.sub(omit_body, text)
    text = _LONG_BASE64_RE.sub(
        lambda match: f"<omitted base64 chars={len(match.group(0))}>",
        text,
    )
    limit = max(256, int(max_chars))
    if len(text) > limit:
        omitted = len(text) - limit
        text = f"{text[:limit]}\n<log truncated chars={omitted}>"
    return text


def summarize_subprocess_error(stdout: object, stderr: object) -> str:
    """Extract a compact error from a JSON-printing child process."""
    stderr_text = str(stderr or "").strip()
    if stderr_text:
        return sanitize_log_text(stderr_text, max_chars=2_000)

    stdout_text = str(stdout or "").strip()
    payload: dict[str, Any] | None = None
    for line in reversed(stdout_text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break

    if payload is None:
        return sanitize_log_text(stdout_text or "subprocess failed", max_chars=2_000)

    parts: list[str] = []
    error = sanitize_log_text(payload.get("error", ""), max_chars=1_000).strip()
    parts.append(error or "subprocess failed")

    event_stderr = sanitize_log_text(payload.get("event_stderr", ""), max_chars=500).strip()
    if event_stderr and event_stderr not in parts:
        parts.append(f"event_stderr={event_stderr}")

    sd_result = summarize_sd_prompt_result(payload.get("sd_prompt_send_result"))
    if sd_result:
        status = int(sd_result.get("status", 0) or 0)
        sd_error = sanitize_log_text(sd_result.get("error", ""), max_chars=300).strip()
        body_chars = int(sd_result.get("body_chars", 0) or 0)
        summary = f"sd_status={status}"
        if sd_error:
            summary += f" sd_error={sd_error}"
        if body_chars:
            summary += f" body_omitted={body_chars}chars"
        parts.append(summary)

    return sanitize_log_text("; ".join(parts), max_chars=2_000)
