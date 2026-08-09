from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO
import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile

from scripts.j_paths import data_path


DEFAULT_DB_PATH = data_path("grok_export_index", "grok_export_2026-07-11.sqlite")
DEFAULT_SOURCES_DIR = data_path("grok_export_index", "sources")
GROK_JSON_SUFFIX = "prod-grok-backend.json"
SCHEMA_VERSION = 2
IMPORTER_VERSION = "grok-importer/2"
UUID_RE = re.compile(
    r"(?i)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_archives (
    sha256 TEXT PRIMARY KEY,
    original_path TEXT NOT NULL,
    canonical_path TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    zip_entry_count INTEGER,
    grok_json_entry TEXT,
    first_seen_at TEXT NOT NULL,
    imported_at TEXT,
    status TEXT NOT NULL,
    error TEXT,
    schema_version INTEGER NOT NULL,
    importer_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    archive_sha256 TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    conversations_seen INTEGER NOT NULL DEFAULT 0,
    messages_seen INTEGER NOT NULL DEFAULT 0,
    attachments_seen INTEGER NOT NULL DEFAULT 0,
    projects_seen INTEGER NOT NULL DEFAULT 0,
    media_posts_seen INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    schema_version INTEGER NOT NULL,
    importer_version TEXT NOT NULL,
    FOREIGN KEY(archive_sha256) REFERENCES source_archives(sha256)
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    create_time REAL,
    update_time REAL,
    create_iso TEXT,
    update_iso TEXT,
    source_file TEXT,
    current_node TEXT,
    default_model_slug TEXT,
    temporary INTEGER NOT NULL DEFAULT 0,
    starred INTEGER NOT NULL DEFAULT 0,
    summary TEXT,
    raw_json TEXT NOT NULL,
    archive_sha256 TEXT NOT NULL,
    FOREIGN KEY(archive_sha256) REFERENCES source_archives(sha256)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    node_id TEXT NOT NULL UNIQUE,
    parent_id TEXT,
    role TEXT NOT NULL,
    author_name TEXT,
    create_time REAL,
    update_time REAL,
    create_iso TEXT,
    content_type TEXT,
    text TEXT,
    status TEXT,
    model_slug TEXT,
    source_file TEXT,
    raw_json TEXT NOT NULL,
    archive_sha256 TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id),
    FOREIGN KEY(archive_sha256) REFERENCES source_archives(sha256)
);

CREATE TABLE IF NOT EXISTS response_payloads (
    message_id INTEGER PRIMARY KEY,
    response_id TEXT NOT NULL UNIQUE,
    metadata_json TEXT,
    thinking_json TEXT,
    web_search_json TEXT,
    file_attachments_json TEXT,
    card_attachments_json TEXT,
    payload_json TEXT NOT NULL,
    archive_sha256 TEXT NOT NULL,
    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY(archive_sha256) REFERENCES source_archives(sha256)
);

CREATE TABLE IF NOT EXISTS response_share_links (
    message_id INTEGER PRIMARY KEY,
    response_id TEXT NOT NULL UNIQUE,
    share_link_id TEXT,
    conversation_id TEXT,
    create_iso TEXT,
    is_public INTEGER,
    view_count INTEGER,
    raw_json TEXT NOT NULL,
    archive_sha256 TEXT NOT NULL,
    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY(archive_sha256) REFERENCES source_archives(sha256)
);

CREATE TABLE IF NOT EXISTS attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    archive_sha256 TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    zip_entry TEXT,
    present_in_archive INTEGER NOT NULL,
    file_size INTEGER,
    compressed_size INTEGER,
    crc32 TEXT,
    sha256 TEXT,
    mime_type TEXT,
    media_kind TEXT,
    reference_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(archive_sha256) REFERENCES source_archives(sha256)
);

CREATE TABLE IF NOT EXISTS message_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    attachment_id INTEGER NOT NULL,
    asset_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    raw_reference TEXT,
    UNIQUE(message_id, relation_type, ordinal, attachment_id),
    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY(attachment_id) REFERENCES attachments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT,
    create_iso TEXT,
    last_use_iso TEXT,
    kind TEXT,
    raw_json TEXT NOT NULL,
    archive_sha256 TEXT NOT NULL,
    FOREIGN KEY(archive_sha256) REFERENCES source_archives(sha256)
);

CREATE TABLE IF NOT EXISTS media_posts (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    original_prompt TEXT,
    media_type TEXT,
    create_time REAL,
    create_iso TEXT,
    link TEXT,
    raw_json TEXT NOT NULL,
    archive_sha256 TEXT NOT NULL,
    FOREIGN KEY(archive_sha256) REFERENCES source_archives(sha256)
);

CREATE TABLE IF NOT EXISTS media_post_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_post_id TEXT NOT NULL,
    attachment_id INTEGER NOT NULL,
    asset_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    UNIQUE(media_post_id, ordinal, attachment_id),
    FOREIGN KEY(media_post_id) REFERENCES media_posts(id) ON DELETE CASCADE,
    FOREIGN KEY(attachment_id) REFERENCES attachments(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    title,
    role,
    text,
    conversation_id UNINDEXED,
    node_id UNINDEXED,
    tokenize='unicode61'
);

CREATE INDEX IF NOT EXISTS idx_conversations_update ON conversations(update_time);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_attachments_present ON attachments(present_in_archive);
CREATE UNIQUE INDEX IF NOT EXISTS idx_attachments_physical_member
    ON attachments(archive_sha256, zip_entry) WHERE zip_entry IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_attachments_missing_asset
    ON attachments(archive_sha256, asset_id) WHERE present_in_archive = 0;
CREATE INDEX IF NOT EXISTS idx_attachments_asset
    ON attachments(archive_sha256, asset_id);
CREATE INDEX IF NOT EXISTS idx_message_attachments_asset ON message_attachments(asset_id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def existing_database_version(path: Path) -> tuple[int, str] | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        has_metadata = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_metadata'"
        ).fetchone()
        if not has_metadata:
            return (1, "grok-importer/1")
        values = dict(conn.execute("SELECT key,value FROM schema_metadata"))
        return (int(values.get("schema_version") or 1), values.get("importer_version") or "unknown")
    finally:
        conn.close()


def backup_and_reset_database(path: Path, old_version: tuple[int, str]) -> Path:
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / (
        f"{path.stem}.schema-v{old_version[0]}-to-v{SCHEMA_VERSION}.{stamp}.sqlite"
    )
    source = sqlite3.connect(path)
    destination = sqlite3.connect(backup_path)
    try:
        source.execute("PRAGMA wal_checkpoint(FULL)")
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            candidate.unlink()
    print(
        f"database_rebuild old_schema={old_version[0]} old_importer={old_version[1]} "
        f"backup={backup_path}",
        flush=True,
    )
    return backup_path


def prepare_database(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_version = existing_database_version(path)
    if old_version is None:
        return {"rebuilt": False, "backup_path": "", "old_version": None}
    if old_version == (SCHEMA_VERSION, IMPORTER_VERSION):
        return {"rebuilt": False, "backup_path": "", "old_version": old_version}
    backup = backup_and_reset_database(path, old_version)
    return {"rebuilt": True, "backup_path": str(backup), "old_version": old_version}


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=120)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.executemany(
        "INSERT OR IGNORE INTO schema_metadata(key,value) VALUES (?, ?)",
        (
            ("schema_version", str(SCHEMA_VERSION)),
            ("importer_version", IMPORTER_VERSION),
        ),
    )
    conn.commit()
    return conn


def sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def stage_source(
    conn: sqlite3.Connection,
    source: Path,
    sources_dir: Path,
) -> tuple[str, Path, bool, bool]:
    stat = source.stat()
    original = str(source.resolve())
    sources_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="grok-export-", suffix=".partial", dir=sources_dir)
    os.close(fd)
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    canonical_repaired = False
    try:
        with source.open("rb") as src, temp_path.open("wb") as dst:
            copied = 0
            while True:
                block = src.read(4 * 1024 * 1024)
                if not block:
                    break
                dst.write(block)
                digest.update(block)
                copied += len(block)
                if copied % (128 * 1024 * 1024) < len(block):
                    print(f"copy_progress bytes={copied} total={stat.st_size}", flush=True)
            dst.flush()
            os.fsync(dst.fileno())
        if copied != stat.st_size:
            raise RuntimeError(
                f"source size changed while staging: expected={stat.st_size} copied={copied}"
            )
        sha256 = digest.hexdigest()
        with temp_path.open("rb") as handle:
            temp_sha256 = sha256_stream(handle)
        if temp_sha256 != sha256:
            raise RuntimeError(
                f"staged source hash mismatch: source={sha256} staged={temp_sha256}"
            )
        canonical = sources_dir / f"{sha256}.zip"
        if canonical.exists():
            with canonical.open("rb") as handle:
                canonical_sha256 = sha256_stream(handle)
            if canonical_sha256 == sha256:
                temp_path.unlink()
            else:
                os.replace(temp_path, canonical)
                canonical_repaired = True
                with canonical.open("rb") as handle:
                    canonical_sha256 = sha256_stream(handle)
        else:
            os.replace(temp_path, canonical)
            with canonical.open("rb") as handle:
                canonical_sha256 = sha256_stream(handle)
        if canonical_sha256 != sha256:
            raise RuntimeError(
                f"canonical source hash mismatch: source={sha256} canonical={canonical_sha256}"
            )
    finally:
        if temp_path.exists():
            temp_path.unlink()

    known = conn.execute(
        """
        SELECT original_path, canonical_path, size_bytes, mtime_ns, status,
               schema_version, importer_version
        FROM source_archives WHERE sha256=?
        """,
        (sha256,),
    ).fetchone()
    metadata_unchanged = known and (
        str(known["original_path"]) == original
        and str(known["canonical_path"]) == str(canonical)
        and int(known["size_bytes"]) == stat.st_size
        and int(known["mtime_ns"]) == stat.st_mtime_ns
        and int(known["schema_version"]) == SCHEMA_VERSION
        and str(known["importer_version"]) == IMPORTER_VERSION
    )
    if metadata_unchanged:
        return sha256, canonical, str(known["status"]) == "imported", canonical_repaired

    with conn:
        conn.execute(
            """
            INSERT INTO source_archives(
                sha256, original_path, canonical_path, size_bytes, mtime_ns,
                first_seen_at, status, schema_version, importer_version
            ) VALUES (?, ?, ?, ?, ?, ?, 'staged', ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
                original_path=excluded.original_path,
                canonical_path=excluded.canonical_path,
                size_bytes=excluded.size_bytes,
                mtime_ns=excluded.mtime_ns,
                status=CASE WHEN source_archives.status='imported' THEN 'imported' ELSE 'staged' END,
                error=NULL,
                schema_version=excluded.schema_version,
                importer_version=excluded.importer_version
            """,
            (
                sha256, original, str(canonical), stat.st_size, stat.st_mtime_ns,
                utc_now(), SCHEMA_VERSION, IMPORTER_VERSION,
            ),
        )
    return sha256, canonical, bool(known and str(known["status"]) == "imported"), canonical_repaired


def extended_time(value: Any) -> tuple[float | None, str]:
    candidate = value
    if isinstance(candidate, dict) and "$date" in candidate:
        candidate = candidate["$date"]
    if isinstance(candidate, dict) and "$numberLong" in candidate:
        candidate = candidate["$numberLong"]
    if candidate in (None, ""):
        return None, ""
    try:
        seconds = float(candidate) / 1000.0
        iso = datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")
        return seconds, iso
    except (TypeError, ValueError, OSError):
        return None, str(candidate)


def iso_time(value: Any) -> tuple[float | None, str]:
    text = str(value or "")
    if not text:
        return None, ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.timestamp(), text
    except ValueError:
        return None, text


def asset_id_for_entry(name: str) -> str:
    clean = name.rstrip("/")
    parts = clean.split("/")
    if parts[-1] in {"content", "thumbnail"} and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def detect_media_type(name: str, head: bytes, size: int) -> tuple[str, str]:
    if size == 0:
        return "application/x-empty", "empty"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "image"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "image"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp", "image"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", "image"
    if head.startswith(b"%PDF"):
        return "application/pdf", "document"
    lower = name.lower()
    if lower.endswith(".webp"):
        return "image/webp", "image"
    stripped = head.lstrip(b"\xef\xbb\xbf\r\n\t ")
    if stripped.startswith((b"{", b"[")):
        return "application/json", "data"
    try:
        head.decode("utf-8")
        return "text/plain", "text"
    except UnicodeDecodeError:
        return "application/octet-stream", "binary"


def hash_attachment(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, str, str]:
    digest = hashlib.sha256()
    head = b""
    with zf.open(info) as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            if len(head) < 4096:
                head += block[: 4096 - len(head)]
            digest.update(block)
    mime, kind = detect_media_type(info.filename, head, info.file_size)
    return digest.hexdigest(), mime, kind


def normalize_role(sender: Any) -> str:
    return "user" if str(sender or "").lower() == "human" else "assistant"


def response_status(response: dict[str, Any]) -> str:
    if response.get("error"):
        return "error"
    if response.get("partial"):
        return "partial"
    return "complete"


def referenced_ids(value: Any) -> list[str]:
    return UUID_RE.findall(json_text(value))


def import_payload(
    conn: sqlite3.Connection,
    archive_sha256: str,
    canonical: Path,
) -> dict[str, Any]:
    completed = conn.execute(
        """
        SELECT id FROM import_runs
        WHERE archive_sha256=? AND status='completed'
          AND schema_version=? AND importer_version=?
        ORDER BY id DESC LIMIT 1
        """,
        (archive_sha256, SCHEMA_VERSION, IMPORTER_VERSION),
    ).fetchone()
    if completed:
        return {"status": "skipped", "archive_sha256": archive_sha256, "run_id": int(completed["id"])}

    started = utc_now()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO import_runs(
                archive_sha256, started_at, status, schema_version, importer_version
            ) VALUES (?, ?, 'running', ?, ?)
            """,
            (archive_sha256, started, SCHEMA_VERSION, IMPORTER_VERSION),
        )
        run_id = int(cursor.lastrowid)

    try:
        with zipfile.ZipFile(canonical) as zf:
            names = zf.namelist()
            grok_entries = [name for name in names if name.endswith(GROK_JSON_SUFFIX)]
            if len(grok_entries) != 1:
                raise RuntimeError(f"expected one {GROK_JSON_SUFFIX}, found {len(grok_entries)}")
            grok_entry = grok_entries[0]
            with zf.open(grok_entry) as handle:
                payload = json.load(handle)

            conversations = list(payload.get("conversations") or [])
            projects = list(payload.get("projects") or [])
            media_posts = list(payload.get("media_posts") or [])
            response_total = sum(len(item.get("responses") or []) for item in conversations)
            print(
                f"import_start conversations={len(conversations)} responses={response_total} "
                f"projects={len(projects)} media_posts={len(media_posts)}",
                flush=True,
            )

            asset_infos = [
                info for info in zf.infolist()
                if not info.is_dir() and not info.filename.lower().endswith(".json")
            ]
            actual_asset_ids: set[str] = set()
            attachment_rows: list[tuple[Any, ...]] = []
            for index, info in enumerate(asset_infos, 1):
                asset_id = asset_id_for_entry(info.filename)
                actual_asset_ids.add(asset_id)
                file_sha, mime, kind = hash_attachment(zf, info)
                attachment_rows.append(
                    (
                        archive_sha256, asset_id, info.filename, 1, info.file_size,
                        info.compress_size, f"{info.CRC:08x}", file_sha, mime, kind,
                    )
                )
                if index % 50 == 0 or index == len(asset_infos):
                    print(f"attachment_progress hashed={index} total={len(asset_infos)}", flush=True)

        source_file = f"{canonical}!/{grok_entry}"
        message_links: list[tuple[str, str, int, str]] = []
        media_links: list[tuple[str, int, str]] = []
        referenced_missing: set[str] = set()
        counts = Counter()

        with conn:
            conn.executemany(
                """
                INSERT INTO attachments(
                    archive_sha256, asset_id, zip_entry, present_in_archive,
                    file_size, compressed_size, crc32, sha256, mime_type, media_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                attachment_rows,
            )

            for item in conversations:
                meta = dict(item.get("conversation") or {})
                conversation_id = str(meta.get("id") or "")
                if not conversation_id:
                    raise RuntimeError("conversation without id")
                create_time, create_iso = iso_time(meta.get("create_time"))
                update_time, update_iso = iso_time(meta.get("modify_time"))
                wrappers = [dict(wrapped or {}) for wrapped in item.get("responses") or []]
                responses = [dict(wrapped.get("response") or {}) for wrapped in wrappers]
                models = Counter(str(r.get("model") or "") for r in responses if r.get("model"))
                default_model = models.most_common(1)[0][0] if models else ""
                conn.execute(
                    """
                    INSERT INTO conversations(
                        id, title, create_time, update_time, create_iso, update_iso,
                        source_file, current_node, default_model_slug, temporary,
                        starred, summary, raw_json, archive_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title, create_time=excluded.create_time,
                        update_time=excluded.update_time, create_iso=excluded.create_iso,
                        update_iso=excluded.update_iso, source_file=excluded.source_file,
                        current_node=excluded.current_node,
                        default_model_slug=excluded.default_model_slug,
                        temporary=excluded.temporary, starred=excluded.starred,
                        summary=excluded.summary, raw_json=excluded.raw_json,
                        archive_sha256=excluded.archive_sha256
                    """,
                    (
                        conversation_id, str(meta.get("title") or ""), create_time,
                        update_time, create_iso, update_iso, source_file,
                        str(meta.get("leaf_response_id") or ""), default_model,
                        int(bool(meta.get("temporary"))), int(bool(meta.get("starred"))),
                        str(meta.get("summary") or ""), json_text(meta), archive_sha256,
                    ),
                )
                counts["conversations"] += 1

                for wrapped, response in zip(wrappers, responses):
                    response_id = str(response.get("_id") or "")
                    if not response_id:
                        raise RuntimeError(f"response without id in conversation={conversation_id}")
                    create_seconds, response_iso = extended_time(response.get("create_time"))
                    text = str(response.get("message") or "")
                    role = normalize_role(response.get("sender"))
                    media_types = list(response.get("media_types") or [])
                    content_type = "multimodal" if media_types else ("text" if text.strip() else "empty")
                    conn.execute(
                        """
                        INSERT INTO messages(
                            conversation_id, node_id, parent_id, role, author_name,
                            create_time, update_time, create_iso, content_type, text,
                            status, model_slug, source_file, raw_json, archive_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(node_id) DO UPDATE SET
                            conversation_id=excluded.conversation_id,
                            parent_id=excluded.parent_id, role=excluded.role,
                            author_name=excluded.author_name, create_time=excluded.create_time,
                            update_time=excluded.update_time, create_iso=excluded.create_iso,
                            content_type=excluded.content_type, text=excluded.text,
                            status=excluded.status, model_slug=excluded.model_slug,
                            source_file=excluded.source_file, raw_json=excluded.raw_json,
                            archive_sha256=excluded.archive_sha256
                        """,
                        (
                            conversation_id, response_id,
                            str(response.get("parent_response_id") or ""), role,
                            str(response.get("sender") or ""), create_seconds,
                            create_seconds, response_iso, content_type, text,
                            response_status(response), str(response.get("model") or ""),
                            source_file, json_text(response), archive_sha256,
                        ),
                    )
                    row = conn.execute("SELECT id FROM messages WHERE node_id=?", (response_id,)).fetchone()
                    message_id = int(row["id"])
                    conn.execute("DELETE FROM message_attachments WHERE message_id=?", (message_id,))
                    conn.execute("DELETE FROM response_share_links WHERE message_id=?", (message_id,))
                    share_link = wrapped.get("share_link")
                    if share_link:
                        share = dict(share_link)
                        conn.execute(
                            """
                            INSERT INTO response_share_links(
                                message_id,response_id,share_link_id,conversation_id,
                                create_iso,is_public,view_count,raw_json,archive_sha256
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(message_id) DO UPDATE SET
                                response_id=excluded.response_id,
                                share_link_id=excluded.share_link_id,
                                conversation_id=excluded.conversation_id,
                                create_iso=excluded.create_iso,
                                is_public=excluded.is_public,
                                view_count=excluded.view_count,
                                raw_json=excluded.raw_json,
                                archive_sha256=excluded.archive_sha256
                            """,
                            (
                                message_id, response_id, str(share.get("share_link_id") or ""),
                                str(share.get("conversation_id") or conversation_id),
                                str(share.get("create_time") or ""),
                                int(bool(share.get("is_public"))), int(share.get("view_count") or 0),
                                json_text(share_link), archive_sha256,
                            ),
                        )
                    thinking = {
                        key: response.get(key) for key in (
                            "thinking_start_time", "thinking_end_time", "thinking_trace",
                            "agent_thinking_traces", "steps",
                        ) if key in response
                    }
                    conn.execute(
                        """
                        INSERT INTO response_payloads(
                            message_id, response_id, metadata_json, thinking_json,
                            web_search_json, file_attachments_json,
                            card_attachments_json, payload_json, archive_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(message_id) DO UPDATE SET
                            response_id=excluded.response_id,
                            metadata_json=excluded.metadata_json,
                            thinking_json=excluded.thinking_json,
                            web_search_json=excluded.web_search_json,
                            file_attachments_json=excluded.file_attachments_json,
                            card_attachments_json=excluded.card_attachments_json,
                            payload_json=excluded.payload_json,
                            archive_sha256=excluded.archive_sha256
                        """,
                        (
                            message_id, response_id, json_text(response.get("metadata") or {}),
                            json_text(thinking), json_text(response.get("web_search_results") or []),
                            json_text(response.get("file_attachments") or []),
                            json_text(response.get("card_attachments_json") or []),
                            json_text(response), archive_sha256,
                        ),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO messages_fts(rowid,title,role,text,conversation_id,node_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (message_id, str(meta.get("title") or ""), role, text, conversation_id, response_id),
                    )
                    direct = [str(value) for value in response.get("file_attachments") or [] if value]
                    seen: set[tuple[str, str]] = set()
                    ordinal = 0
                    for asset_id in direct:
                        if asset_id not in actual_asset_ids:
                            referenced_missing.add(asset_id)
                        message_links.append((response_id, "file_attachment", ordinal, asset_id))
                        seen.add(("file_attachment", asset_id))
                        ordinal += 1
                    embedded = [value for value in referenced_ids(response) if value in actual_asset_ids]
                    for asset_id in dict.fromkeys(embedded):
                        if ("file_attachment", asset_id) in seen:
                            continue
                        message_links.append((response_id, "embedded_reference", ordinal, asset_id))
                        ordinal += 1
                    counts[role] += 1
                    counts["messages"] += 1

            for index, project in enumerate(projects):
                project_id = str(project.get("workspace_id") or f"project-{archive_sha256[:12]}-{index}")
                conn.execute(
                    """
                    INSERT INTO projects(id,name,create_iso,last_use_iso,kind,raw_json,archive_sha256)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name, create_iso=excluded.create_iso,
                        last_use_iso=excluded.last_use_iso, kind=excluded.kind,
                        raw_json=excluded.raw_json, archive_sha256=excluded.archive_sha256
                    """,
                    (
                        project_id, str(project.get("name") or ""),
                        str(project.get("create_time") or ""), str(project.get("last_use_time") or ""),
                        str(project.get("kind") or ""), json_text(project), archive_sha256,
                    ),
                )

            for post in media_posts:
                post_id = str(post.get("id") or "")
                if not post_id:
                    continue
                conn.execute(
                    "DELETE FROM media_post_attachments WHERE media_post_id=?",
                    (post_id,),
                )
                create_seconds, create_iso = extended_time(post.get("create_time"))
                conn.execute(
                    """
                    INSERT INTO media_posts(
                        id,user_id,original_prompt,media_type,create_time,create_iso,
                        link,raw_json,archive_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        user_id=excluded.user_id, original_prompt=excluded.original_prompt,
                        media_type=excluded.media_type, create_time=excluded.create_time,
                        create_iso=excluded.create_iso, link=excluded.link,
                        raw_json=excluded.raw_json, archive_sha256=excluded.archive_sha256
                    """,
                    (
                        post_id, str(post.get("user_id") or ""),
                        str(post.get("original_prompt") or ""), str(post.get("media_type") or ""),
                        create_seconds, create_iso, str(post.get("link") or ""),
                        json_text(post), archive_sha256,
                    ),
                )
                if post_id not in actual_asset_ids:
                    referenced_missing.add(post_id)
                media_links.append((post_id, 0, post_id))

            conn.executemany(
                """
                INSERT OR IGNORE INTO attachments(
                    archive_sha256,asset_id,zip_entry,present_in_archive,media_kind
                ) VALUES (?, ?, NULL, 0, 'missing')
                """,
                [(archive_sha256, asset_id) for asset_id in sorted(referenced_missing)],
            )
            for response_id, relation, ordinal, asset_id in message_links:
                row = conn.execute("SELECT id FROM messages WHERE node_id=?", (response_id,)).fetchone()
                attachment_ids = conn.execute(
                    """
                    SELECT id FROM attachments
                    WHERE archive_sha256=? AND asset_id=?
                    ORDER BY present_in_archive DESC, zip_entry
                    """,
                    (archive_sha256, asset_id),
                ).fetchall()
                for attachment in attachment_ids:
                    conn.execute(
                        """
                        INSERT INTO message_attachments(
                            message_id,attachment_id,asset_id,relation_type,ordinal,raw_reference
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(row["id"]), int(attachment["id"]), asset_id,
                            relation, ordinal, asset_id,
                        ),
                    )
            for post_id, ordinal, asset_id in media_links:
                attachment_ids = conn.execute(
                    """
                    SELECT id FROM attachments
                    WHERE archive_sha256=? AND asset_id=?
                    ORDER BY present_in_archive DESC, zip_entry
                    """,
                    (archive_sha256, asset_id),
                ).fetchall()
                for attachment in attachment_ids:
                    conn.execute(
                        """
                        INSERT INTO media_post_attachments(
                            media_post_id,attachment_id,asset_id,ordinal
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (post_id, int(attachment["id"]), asset_id, ordinal),
                    )
            conn.execute(
                """
                UPDATE attachments SET reference_count =
                    (SELECT COUNT(*) FROM message_attachments ma
                     WHERE ma.attachment_id=attachments.id)
                  + (SELECT COUNT(*) FROM media_post_attachments mpa
                     WHERE mpa.attachment_id=attachments.id)
                """,
            )
            completed_at = utc_now()
            conn.execute(
                """
                UPDATE source_archives SET
                    zip_entry_count=?, grok_json_entry=?, imported_at=?, status='imported',
                    error=NULL, schema_version=?, importer_version=?
                WHERE sha256=?
                """,
                (
                    len(names), grok_entry, completed_at, SCHEMA_VERSION,
                    IMPORTER_VERSION, archive_sha256,
                ),
            )
            conn.execute(
                """
                UPDATE import_runs SET
                    completed_at=?, status='completed', conversations_seen=?, messages_seen=?,
                    attachments_seen=?, projects_seen=?, media_posts_seen=?
                WHERE id=?
                """,
                (
                    completed_at, len(conversations), response_total,
                    len(asset_infos), len(projects), len(media_posts), run_id,
                ),
            )

        stored_counts = {
            "share_links": int(conn.execute("SELECT COUNT(*) FROM response_share_links").fetchone()[0]),
            "attachments_missing": int(
                conn.execute("SELECT COUNT(*) FROM attachments WHERE present_in_archive=0").fetchone()[0]
            ),
            "message_attachment_links": int(conn.execute("SELECT COUNT(*) FROM message_attachments").fetchone()[0]),
            "media_post_attachment_links": int(
                conn.execute("SELECT COUNT(*) FROM media_post_attachments").fetchone()[0]
            ),
        }
        return {
            "status": "imported",
            "schema_version": SCHEMA_VERSION,
            "importer_version": IMPORTER_VERSION,
            "archive_sha256": archive_sha256,
            "run_id": run_id,
            "conversations": counts["conversations"],
            "messages": counts["messages"],
            "roles": {"user": counts["user"], "assistant": counts["assistant"]},
            "attachments_present": len(asset_infos),
            **stored_counts,
            "projects": len(projects),
            "media_posts": len(media_posts),
        }
    except Exception as exc:
        with conn:
            conn.execute(
                "UPDATE import_runs SET completed_at=?, status='failed', error=? WHERE id=?",
                (utc_now(), str(exc), run_id),
            )
            conn.execute(
                "UPDATE source_archives SET status='failed', error=? WHERE sha256=?",
                (str(exc), archive_sha256),
            )
        raise


def import_grok_archive(
    source_zip: str | Path,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    sources_dir: str | Path = DEFAULT_SOURCES_DIR,
) -> dict[str, Any]:
    source = Path(source_zip)
    if not source.is_file():
        raise FileNotFoundError(f"Grok export ZIP not found: {source}")
    migration = prepare_database(db_path)
    conn = connect(db_path)
    try:
        archive_sha256, canonical, source_reused, canonical_repaired = stage_source(
            conn, source, Path(sources_dir)
        )
        result = import_payload(conn, archive_sha256, canonical)
        result.update(
            {
                "source_zip": str(source.resolve()),
                "canonical_zip": str(canonical),
                "db_path": str(Path(db_path)),
                "source_reused": source_reused,
                "canonical_repaired": canonical_repaired,
                "schema_version": SCHEMA_VERSION,
                "importer_version": IMPORTER_VERSION,
                "database_rebuilt": migration["rebuilt"],
                "backup_path": migration["backup_path"],
            }
        )
        return result
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grok export ZIPを検索互換SQLiteへ取り込む")
    parser.add_argument("source_zip")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--sources-dir", default=str(DEFAULT_SOURCES_DIR))
    args = parser.parse_args()
    result = import_grok_archive(args.source_zip, db_path=args.db, sources_dir=args.sources_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
