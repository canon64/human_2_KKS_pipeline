from __future__ import annotations

import os
from pathlib import Path


def save_env_secret(env_file: str, key: str, value: str) -> Path:
    """指定キーだけを.envへ保存し、他の行は維持する。"""
    path = Path(env_file).expanduser()
    name = str(key or "").strip()
    secret = str(value or "").strip()
    if not name or not name.replace("_", "A").isalnum() or name[0].isdigit():
        raise ValueError("環境変数名が正しくない")
    if not secret:
        raise ValueError("Botトークンが空")

    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    replacement = f"{name}={secret}"
    updated: list[str] = []
    replaced = False
    for line in lines:
        current = line.strip()
        if current and not current.startswith("#") and current.partition("=")[0].strip() == name:
            if not replaced:
                updated.append(replacement)
                replaced = True
            continue
        updated.append(line)
    if not replaced:
        updated.append(replacement)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temp.write_text("\n".join(updated) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return path
