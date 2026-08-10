"""Discord Bot のユーザープロフィールを更新する。"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DISCORD_CURRENT_USER_URL = "https://discord.com/api/v10/users/@me"
MAX_AVATAR_BYTES = 10 * 1024 * 1024


def _load_env_value(env_file: str, key: str) -> str:
    value = os.environ.get(key, "").strip()
    if value:
        return value
    path = Path(env_file)
    if not env_file or not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, raw_value = line.partition("=")
        if name.strip() == key:
            return raw_value.strip().strip('"').strip("'")
    return ""


def _avatar_data_uri(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"アイコン画像が見つかりません: {path}")
    data = path.read_bytes()
    if not data:
        raise ValueError("アイコン画像が空です")
    if len(data) > MAX_AVATAR_BYTES:
        raise ValueError("アイコン画像は10MB以下にしてください")
    mime = mimetypes.guess_type(path.name)[0] or ""
    if mime not in {"image/png", "image/jpeg", "image/gif"}:
        raise ValueError("アイコン画像はPNG、JPEG、GIFのいずれかを指定してください")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def update_bot_profile(
    *,
    token_env: str,
    env_file: str,
    username: str = "",
    avatar_path: str = "",
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """Bot名・アイコンの指定された方だけを Discord へ反映する。"""
    token_key = token_env.strip() or "DISCORD_BOT_TOKEN"
    token = _load_env_value(env_file, token_key)
    if not token:
        raise ValueError(f"Botトークンが環境変数 {token_key} にありません")

    payload: dict[str, str] = {}
    clean_name = username.strip()
    clean_avatar = avatar_path.strip()
    if clean_name:
        if not 2 <= len(clean_name) <= 32:
            raise ValueError("Bot名は2～32文字で指定してください")
        payload["username"] = clean_name
    if clean_avatar:
        payload["avatar"] = _avatar_data_uri(clean_avatar)
    if not payload:
        raise ValueError("Bot名またはアイコン画像を指定してください")

    request = urllib.request.Request(
        DISCORD_CURRENT_USER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "human-2-kks-pipeline/1.0",
        },
        method="PATCH",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("message", body)
        except Exception:
            detail = body
        raise RuntimeError(f"Discord API {exc.code}: {detail}") from exc
