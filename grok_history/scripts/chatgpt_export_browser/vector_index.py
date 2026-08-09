from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import re
import sqlite3

from .core import DEFAULT_DB_PATH
from scripts.j_paths import data_path


DEFAULT_VECTOR_DB_PATH = data_path("chatgpt_export_index", "chatgpt_vectors_2026-07-04.sqlite")
CODE_FENCE_RE = re.compile(r"```([A-Za-z0-9_+.#-]*)[^\n]*\n(.*?)```", re.DOTALL)
SHORT_TEXT_CHARS = 80
TEXT_CHUNK_CHARS = 4200
TEXT_OVERLAP_CHARS = 500
CODE_CHUNK_LINES = 140
CODE_OVERLAP_LINES = 20
MIN_STANDALONE_CHARS = 8
SMALL_PRICE_PER_MILLION = 0.02
LARGE_PRICE_PER_MILLION = 0.13


@dataclass(frozen=True)
class RawMessage:
    conversation_id: str
    message_id: int
    node_id: str
    title: str
    role: str
    create_iso: str
    source_file: str
    text: str


@dataclass(frozen=True)
class Chunk:
    conversation_id: str
    message_id: int
    node_id: str
    title: str
    role: str
    create_iso: str
    source_file: str
    chunk_index: int
    chunk_kind: str
    language: str
    token_estimate: int
    text: str
    embedding_input: str


def estimate_tokens(text: str) -> int:
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    non_ascii_count = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4 + non_ascii_count / 1.6))


def embedding_text(chunk: Chunk) -> str:
    header = [
        f"title: {chunk.title}",
        f"role: {chunk.role}",
        f"kind: {chunk.chunk_kind}",
    ]
    if chunk.language:
        header.append(f"language: {chunk.language}")
    if chunk.create_iso:
        header.append(f"date: {chunk.create_iso[:10]}")
    return "\n".join(header) + "\n\n" + chunk.text.strip()


def is_thoughts_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    content_type = obj.get("content_type")
    return content_type in {"thoughts", "reasoning_recap"} or "thoughts" in obj or "source_analysis_msg_id" in obj


def load_messages(source_db: Path) -> list[RawMessage]:
    with sqlite3.connect(source_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT m.id AS message_id, m.conversation_id, m.node_id, c.title,
                   m.role, m.create_iso, m.source_file, m.text
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE trim(m.text) != ''
            ORDER BY c.update_time DESC, m.id ASC
            """
        ).fetchall()
    return [
        RawMessage(
            conversation_id=row["conversation_id"],
            message_id=int(row["message_id"]),
            node_id=row["node_id"] or "",
            title=row["title"] or "",
            role=row["role"] or "",
            create_iso=row["create_iso"] or "",
            source_file=row["source_file"] or "",
            text=(row["text"] or "").replace("\r\n", "\n").replace("\r", "\n"),
        )
        for row in rows
    ]


def split_text(text: str, target_chars: int = TEXT_CHUNK_CHARS, overlap_chars: int = TEXT_OVERLAP_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= target_chars:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + target_chars)
        if end < len(text):
            break_at = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end), text.rfind("。", start, end))
            if break_at > start + target_chars // 2:
                end = break_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def split_code(code: str) -> list[str]:
    lines = code.strip("\n").splitlines()
    if len(lines) <= CODE_CHUNK_LINES:
        return [code.strip("\n")] if code.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(lines):
        end = min(len(lines), start + CODE_CHUNK_LINES)
        chunk = "\n".join(lines[start:end]).strip("\n")
        if chunk:
            chunks.append(chunk)
        if end >= len(lines):
            break
        start = max(end - CODE_OVERLAP_LINES, start + 1)
    return chunks


def split_message_body(text: str) -> list[tuple[str, str, str]]:
    if is_thoughts_text(text):
        return [("thoughts", "", text.strip())]

    parts: list[tuple[str, str, str]] = []
    cursor = 0
    for match in CODE_FENCE_RE.finditer(text):
        before = text[cursor : match.start()].strip()
        if before:
            parts.extend(("text", "", part) for part in split_text(before))

        language = (match.group(1) or "unknown").strip().lower()
        code = match.group(2)
        parts.extend(("code", language, part) for part in split_code(code))
        cursor = match.end()

    after = text[cursor:].strip()
    if after:
        parts.extend(("text", "", part) for part in split_text(after))

    if not parts:
        return []
    if len(parts) == 1 and parts[0][0] == "text" and CODE_FENCE_RE.search(text) is None:
        return parts
    return parts


def merge_short_messages(messages: list[RawMessage]) -> list[RawMessage]:
    merged: list[RawMessage] = []
    pending: RawMessage | None = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            merged.append(pending)
            pending = None

    for msg in messages:
        text = msg.text.strip()
        if len(text) < MIN_STANDALONE_CHARS:
            continue
        can_merge = (
            pending is not None
            and pending.conversation_id == msg.conversation_id
            and pending.role == msg.role
            and len(pending.text) < SHORT_TEXT_CHARS * 4
            and len(text) < SHORT_TEXT_CHARS
            and not CODE_FENCE_RE.search(text)
            and not is_thoughts_text(text)
        )
        if can_merge:
            pending = RawMessage(
                conversation_id=pending.conversation_id,
                message_id=pending.message_id,
                node_id=pending.node_id,
                title=pending.title,
                role=pending.role,
                create_iso=pending.create_iso,
                source_file=pending.source_file,
                text=pending.text.rstrip() + "\n" + text,
            )
            continue

        flush()
        pending = msg
    flush()
    return merged


def build_chunks(
    messages: list[RawMessage],
    *,
    preserve_source_messages: bool = False,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    selected_messages = messages if preserve_source_messages else merge_short_messages(messages)
    for msg in selected_messages:
        part_index = 0
        for kind, language, text in split_message_body(msg.text):
            if not text.strip():
                continue
            temp = Chunk(
                conversation_id=msg.conversation_id,
                message_id=msg.message_id,
                node_id=msg.node_id,
                title=msg.title,
                role=msg.role,
                create_iso=msg.create_iso,
                source_file=msg.source_file,
                chunk_index=part_index,
                chunk_kind=kind,
                language=language,
                token_estimate=0,
                text=text.strip(),
                embedding_input="",
            )
            prepared = embedding_text(temp)
            chunks.append(
                Chunk(
                    conversation_id=temp.conversation_id,
                    message_id=temp.message_id,
                    node_id=temp.node_id,
                    title=temp.title,
                    role=temp.role,
                    create_iso=temp.create_iso,
                    source_file=temp.source_file,
                    chunk_index=temp.chunk_index,
                    chunk_kind=temp.chunk_kind,
                    language=temp.language,
                    token_estimate=estimate_tokens(prepared),
                    text=temp.text,
                    embedding_input=prepared,
                )
            )
            part_index += 1
    return chunks


def init_vector_db(path: Path, *, replace: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace and path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            message_id INTEGER NOT NULL,
            node_id TEXT,
            title TEXT,
            role TEXT,
            create_iso TEXT,
            source_file TEXT,
            chunk_index INTEGER NOT NULL,
            chunk_kind TEXT NOT NULL,
            language TEXT,
            token_estimate INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding_input TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_conversation ON chunks(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(chunk_kind);
        CREATE INDEX IF NOT EXISTS idx_chunks_role ON chunks(role);
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (chunk_id, model),
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        );
        CREATE TABLE IF NOT EXISTS build_runs (
            id INTEGER PRIMARY KEY,
            source_db TEXT NOT NULL,
            vector_db TEXT NOT NULL,
            created_at TEXT NOT NULL,
            message_count INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            token_estimate INTEGER NOT NULL,
            small_cost_usd REAL NOT NULL,
            large_cost_usd REAL NOT NULL,
            preserve_source_messages INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(build_runs)")}
    if "preserve_source_messages" not in columns:
        conn.execute(
            "ALTER TABLE build_runs ADD COLUMN preserve_source_messages INTEGER NOT NULL DEFAULT 0"
        )
    return conn


def write_chunks(
    source_db: Path,
    vector_db: Path,
    *,
    replace: bool = True,
    preserve_source_messages: bool = False,
) -> dict[str, object]:
    messages = load_messages(source_db)
    chunks = build_chunks(messages, preserve_source_messages=preserve_source_messages)
    total_tokens = sum(chunk.token_estimate for chunk in chunks)
    small_cost = total_tokens / 1_000_000 * SMALL_PRICE_PER_MILLION
    large_cost = total_tokens / 1_000_000 * LARGE_PRICE_PER_MILLION

    conn = init_vector_db(vector_db, replace=replace)
    try:
        with conn:
            if replace:
                conn.execute("DELETE FROM embeddings")
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM build_runs")
            conn.executemany(
                """
                INSERT INTO chunks (
                    conversation_id, message_id, node_id, title, role, create_iso,
                    source_file, chunk_index, chunk_kind, language, token_estimate,
                    text, embedding_input
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.conversation_id,
                        chunk.message_id,
                        chunk.node_id,
                        chunk.title,
                        chunk.role,
                        chunk.create_iso,
                        chunk.source_file,
                        chunk.chunk_index,
                        chunk.chunk_kind,
                        chunk.language,
                        chunk.token_estimate,
                        chunk.text,
                        chunk.embedding_input,
                    )
                    for chunk in chunks
                ],
            )
            conn.execute(
                """
                INSERT INTO build_runs (
                    source_db, vector_db, created_at, message_count, chunk_count,
                    token_estimate, small_cost_usd, large_cost_usd,
                    preserve_source_messages
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(source_db),
                    str(vector_db),
                    datetime.now(timezone.utc).isoformat(),
                    len(messages),
                    len(chunks),
                    total_tokens,
                    small_cost,
                    large_cost,
                    int(preserve_source_messages),
                ),
            )
    finally:
        conn.close()

    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.chunk_kind] = counts.get(chunk.chunk_kind, 0) + 1

    return {
        "source_db": str(source_db),
        "vector_db": str(vector_db),
        "messages": len(messages),
        "chunks": len(chunks),
        "by_kind": counts,
        "token_estimate": total_tokens,
        "small_cost_usd": small_cost,
        "large_cost_usd": large_cost,
        "preserve_source_messages": preserve_source_messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatGPT履歴をベクトル化直前のチャンクDBに変換する")
    parser.add_argument("--source-db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--vector-db", default=str(DEFAULT_VECTOR_DB_PATH))
    parser.add_argument("--keep-existing", action="store_true", help="既存DBを削除せず追記する")
    parser.add_argument(
        "--preserve-source-messages",
        action="store_true",
        help="短文を捨てたり別message_idへ統合せず、各非空source messageを最低1 chunkに保つ",
    )
    args = parser.parse_args()

    result = write_chunks(
        Path(args.source_db),
        Path(args.vector_db),
        replace=not args.keep_existing,
        preserve_source_messages=args.preserve_source_messages,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
