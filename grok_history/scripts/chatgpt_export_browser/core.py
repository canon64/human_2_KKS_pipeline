from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from scripts.j_paths import data_path

DEFAULT_DB_PATH = data_path("chatgpt_export_index", "chatgpt_export_2026-07-04.sqlite")


@dataclass(frozen=True)
class MessageHit:
    message_id: int
    conversation_id: str
    title: str
    conversation_update_iso: str
    role: str
    create_iso: str
    text: str
    snippet: str


@dataclass(frozen=True)
class ConversationSummary:
    conversation_id: str
    title: str
    update_iso: str
    message_count: int


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or DEFAULT_DB_PATH)
    if not path.exists():
        raise FileNotFoundError(f"DBが見つかりません: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def get_stats(db_path: str | Path | None = None) -> dict[str, int | str]:
    with _connect(db_path) as conn:
        conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        latest = conn.execute(
            "SELECT update_iso FROM conversations ORDER BY update_time DESC LIMIT 1"
        ).fetchone()
    return {
        "db_path": str(Path(db_path or DEFAULT_DB_PATH)),
        "conversations": conversations,
        "messages": messages,
        "latest_update_iso": latest[0] if latest else "",
    }


def list_conversations(
    db_path: str | Path | None = None,
    *,
    limit: int | None = None,
) -> list[ConversationSummary]:
    limit_sql = ""
    params: tuple[object, ...] = ()
    if limit is not None:
        limit = max(1, int(limit))
        limit_sql = "LIMIT ?"
        params = (limit,)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT c.id, c.title, c.update_iso, COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.update_time DESC
            {limit_sql}
            """,
            params,
        ).fetchall()
    return [
        ConversationSummary(
            conversation_id=row["id"],
            title=row["title"] or "",
            update_iso=row["update_iso"] or "",
            message_count=int(row["message_count"] or 0),
        )
        for row in rows
    ]


def search_messages(
    query: str,
    db_path: str | Path | None = None,
    *,
    role: str = "",
    limit: int = 100,
) -> list[MessageHit]:
    query = query.strip()
    if not query:
        return []

    limit = max(1, min(int(limit), 10000))
    like_query = f"%{query}%"
    params: list[object] = [like_query]
    role_sql = ""
    if role:
        role_sql = "AND m.role = ?"
        params.append(role)
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT m.id, m.conversation_id, c.title, c.update_iso, m.role,
                   m.create_iso, m.text
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.text LIKE ?
            {role_sql}
            ORDER BY m.id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    return [
        MessageHit(
            message_id=int(row["id"]),
            conversation_id=row["conversation_id"],
            title=row["title"] or "",
            conversation_update_iso=row["update_iso"] or "",
            role=row["role"] or "",
            create_iso=row["create_iso"] or "",
            text=row["text"] or "",
            snippet=make_snippet(row["text"] or "", query),
        )
        for row in rows
    ]


def make_snippet(text: str, query: str, *, before: int = 80, after: int = 160) -> str:
    index = text.find(query)
    if index < 0:
        return text[: before + after].replace("\r\n", "\n").replace("\r", "\n")
    start = max(0, index - before)
    end = min(len(text), index + len(query) + after)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return (prefix + text[start:end] + suffix).replace("\r\n", "\n").replace("\r", "\n")


def get_conversation_messages(
    conversation_id: str,
    db_path: str | Path | None = None,
) -> list[MessageHit]:
    if not conversation_id:
        return []

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.conversation_id, c.title, c.update_iso, m.role,
                   m.create_iso, m.text
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
            ORDER BY m.id ASC
            """,
            (conversation_id,),
        ).fetchall()

    return [
        MessageHit(
            message_id=int(row["id"]),
            conversation_id=row["conversation_id"],
            title=row["title"] or "",
            conversation_update_iso=row["update_iso"] or "",
            role=row["role"] or "",
            create_iso=row["create_iso"] or "",
            text=row["text"] or "",
            snippet="",
        )
        for row in rows
    ]
