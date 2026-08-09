from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any

from scripts.grok_export_browser.ollama_vectors import DEFAULT_RAW_DB


DEFAULT_API = "http://127.0.0.1:8877/search"
DEFAULT_QUERIES = (
    "ゲームの設定を順番に説明するにはどうすればいい？",
    "音声認識した文章を別の処理へ渡したい",
    "データベースを検索しやすく整理したい",
)


def post_query(api_url: str, query: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        api_url,
        data=json.dumps({"query": query, "top_k": 1}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not payload.get("ok"):
        raise RuntimeError(f"API search failed: {payload}")
    return payload


def verify_raw(raw_db: Path, payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("selected result is missing")
    uri = f"file:{raw_db.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute(
            "SELECT id,conversation_id,node_id,role,text FROM messages WHERE id=?",
            (int(selected["message_id"]),),
        ).fetchone()
        if user is None or str(user["role"]) != "user":
            raise RuntimeError("selected message is not a raw DB user message")
        replies = conn.execute(
            """
            SELECT id,role,text FROM messages
            WHERE conversation_id=? AND parent_id=? AND role='assistant'
            ORDER BY create_time,id
            """,
            (str(user["conversation_id"]), str(user["node_id"])),
        ).fetchall()
    raw_assistant = "\n\n".join(str(reply["text"] or "") for reply in replies)
    if str(selected.get("user_text") or "") != str(user["text"] or ""):
        raise RuntimeError("API user_text does not match raw DB")
    if str(selected.get("assistant_text") or "") != raw_assistant:
        raise RuntimeError("API assistant_text does not match raw DB")
    return {
        "query": str(payload.get("query") or ""),
        "score": float(selected.get("score") or 0.0),
        "conversation_id": str(user["conversation_id"]),
        "user_message_id": int(user["id"]),
        "assistant_message_ids": [int(reply["id"]) for reply in replies],
        "user_text_length": len(str(user["text"] or "")),
        "assistant_text_length": len(raw_assistant),
        "user_preview": str(user["text"] or "")[:120],
        "assistant_preview": raw_assistant[:120],
        "raw_db_exact": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Grok history API against raw DB")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--raw-db", type=Path, default=DEFAULT_RAW_DB)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    queries = tuple(args.query) if args.query else DEFAULT_QUERIES
    results = [
        verify_raw(args.raw_db, post_query(args.api, query, args.timeout))
        for query in queries
    ]
    print(json.dumps({"ok": True, "count": len(results), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
