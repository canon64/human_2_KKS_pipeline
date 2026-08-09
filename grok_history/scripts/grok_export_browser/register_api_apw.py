from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path(r"J:\workspaces\agent_process_watch\data\processes.json")
COMMAND_HINT = r"J:\.agents\skills\safe-command\scripts\Start-GrokHistoryResponseApi.ps1"


def _flatten(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and isinstance(item.get("value"), list):
            result.extend(_flatten(item["value"]))
        elif isinstance(item, dict):
            result.append(item)
    return result


def register_api(registry: Path, entry: dict[str, Any]) -> None:
    if registry.exists():
        items = _flatten(json.loads(registry.read_text(encoding="utf-8-sig")))
    else:
        items = []
    pid = int(entry["pid"])
    items = [
        item
        for item in items
        if int(item.get("pid") or 0) != pid
        and str(item.get("command_hint") or "") != COMMAND_HINT
    ]
    items.append(entry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    temp_path = registry.with_name(f"{registry.name}.tmp.{os.getpid()}")
    temp_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, registry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register Grok history response API in APW")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--raw-db", type=Path, required=True)
    parser.add_argument("--vector-db", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    health_url = f"http://127.0.0.1:{args.port}/health"
    search_url = f"http://127.0.0.1:{args.port}/search"
    register_api(
        args.registry,
        {
            "pid": args.pid,
            "name": "Grok履歴応答API",
            "purpose": "音声認識文でGrokユーザー発言を検索し、直接対応するassistant本文を返す",
            "started_by": "Codex via ABC Canvas",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "running",
            "command_hint": COMMAND_HINT,
            "cwd": r"J:\tools\api-scripts\pre_classification",
            "log": str(args.stdout_log),
            "error_log": str(args.stderr_log),
            "port": args.port,
            "health_url": health_url,
            "search_url": search_url,
            "raw_db": str(args.raw_db),
            "vector_db": str(args.vector_db),
            "index_path": str(args.index),
            "dest_path": search_url,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
