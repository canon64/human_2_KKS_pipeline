from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import sqlite3
import zipfile

from scripts.j_paths import data_path

from .import_grok import DEFAULT_DB_PATH


DEFAULT_VECTOR_DB_PATH = data_path("grok_export_index", "grok_vectors_2026-07-11.sqlite")

COUNT_TABLES = (
    "source_archives",
    "import_runs",
    "conversations",
    "messages",
    "response_payloads",
    "response_share_links",
    "attachments",
    "message_attachments",
    "projects",
    "media_posts",
    "media_post_attachments",
    "messages_fts",
)


def verify_database(db_path: str | Path, *, fts_query: str = "grok") -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Grok DB not found: {path}")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        counts = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in COUNT_TABLES
        }
        role_rows = conn.execute("SELECT role, COUNT(*) FROM messages GROUP BY role").fetchall()
        attachment_rows = conn.execute(
            "SELECT present_in_archive, COUNT(*) FROM attachments GROUP BY present_in_archive"
        ).fetchall()
        source = conn.execute(
            """
            SELECT sha256, original_path, canonical_path, size_bytes, zip_entry_count,
                   grok_json_entry, status, schema_version, importer_version
            FROM source_archives
            ORDER BY imported_at DESC LIMIT 1
            """
        ).fetchone()
        samples = conn.execute(
            """
            SELECT c.title, m.role, m.create_iso,
                   replace(replace(substr(m.text, 1, 160), char(13), ' '), char(10), ' ') AS snippet
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE trim(m.text) != ''
            ORDER BY m.create_time DESC, m.id DESC
            LIMIT 2
            """
        ).fetchall()
        forbidden = 0
        forbidden_names = ("prod-mc-auth-mgmt-api.json", "prod-mc-billing.json")
        for table, column in (
            ("conversations", "raw_json"),
            ("messages", "raw_json"),
            ("response_payloads", "payload_json"),
            ("projects", "raw_json"),
            ("media_posts", "raw_json"),
        ):
            for name in forbidden_names:
                forbidden += int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE instr({column}, ?) > 0", (name,)
                    ).fetchone()[0]
                )
        media_presence = dict(
            conn.execute(
                """
                SELECT CASE WHEN a.present_in_archive=1 THEN 'present' ELSE 'missing' END,
                       COUNT(DISTINCT mpa.media_post_id)
                FROM media_post_attachments mpa
                JOIN attachments a ON a.id=mpa.attachment_id
                GROUP BY a.present_in_archive
                """
            )
        )
        metadata = dict(conn.execute("SELECT key,value FROM schema_metadata"))
        original: dict[str, Any] = {}
        if source:
            canonical = Path(str(source["canonical_path"]))
            digest = hashlib.sha256()
            with canonical.open("rb") as handle:
                while True:
                    block = handle.read(4 * 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
            with zipfile.ZipFile(canonical) as zf:
                grok_entry = str(source["grok_json_entry"])
                with zf.open(grok_entry) as handle:
                    payload = json.load(handle)
            wrappers = [
                wrapped
                for item in payload.get("conversations") or []
                for wrapped in item.get("responses") or []
            ]
            original_shares = [wrapped.get("share_link") for wrapped in wrappers if wrapped.get("share_link")]
            stored_shares = [
                json.loads(row[0])
                for row in conn.execute("SELECT raw_json FROM response_share_links ORDER BY response_id")
            ]
            original = {
                "canonical_sha256": digest.hexdigest(),
                "sha256_matches_registry": digest.hexdigest() == str(source["sha256"]),
                "conversations": len(payload.get("conversations") or []),
                "responses": len(wrappers),
                "share_links": len(original_shares),
                "share_links_exact_match": sorted(original_shares, key=json_text_sort_key)
                == sorted(stored_shares, key=json_text_sort_key),
                "media_posts": len(payload.get("media_posts") or []),
            }
        return {
            "db_path": str(path),
            "integrity": conn.execute("PRAGMA integrity_check").fetchone()[0],
            "foreign_key_errors": len(conn.execute("PRAGMA foreign_key_check").fetchall()),
            "counts": counts,
            "roles": {str(row[0]): int(row[1]) for row in role_rows},
            "message_period": list(
                conn.execute("SELECT min(create_iso), max(create_iso) FROM messages").fetchone()
            ),
            "conversation_period": list(
                conn.execute("SELECT min(create_iso), max(update_iso) FROM conversations").fetchone()
            ),
            "attachment_presence": {
                "present" if int(row[0]) else "missing": int(row[1]) for row in attachment_rows
            },
            "media_post_attachment_presence": {
                str(key): int(value) for key, value in media_presence.items()
            },
            "schema_metadata": metadata,
            "fts_query": fts_query,
            "fts_hits": int(
                conn.execute(
                    "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", (fts_query,)
                ).fetchone()[0]
            ),
            "forbidden_auth_billing_entry_hits": forbidden,
            "source": dict(source) if source else {},
            "original_correspondence": original,
            "samples": [dict(row) for row in samples],
        }
    finally:
        conn.close()


def json_text_sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def verify_vector_database(
    vector_db_path: str | Path,
    *,
    source_db_path: str | Path,
) -> dict[str, Any]:
    path = Path(vector_db_path)
    if not path.exists():
        raise FileNotFoundError(f"Grok vector preparation DB not found: {path}")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        build = conn.execute(
            """
            SELECT source_db, vector_db, message_count, chunk_count,
                   token_estimate, small_cost_usd, large_cost_usd,
                   preserve_source_messages
            FROM build_runs ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
        source = sqlite3.connect(f"file:{Path(source_db_path).as_posix()}?mode=ro", uri=True)
        try:
            source_message_ids = {
                int(row[0]) for row in source.execute("SELECT id FROM messages WHERE trim(text) != ''")
            }
            source_conversations = {
                str(row[0])
                for row in source.execute(
                    "SELECT DISTINCT conversation_id FROM messages WHERE trim(text) != ''"
                )
            }
        finally:
            source.close()
        chunk_message_ids = {int(row[0]) for row in conn.execute("SELECT DISTINCT message_id FROM chunks")}
        chunk_conversations = {
            str(row[0]) for row in conn.execute("SELECT DISTINCT conversation_id FROM chunks")
        }
        return {
            "db_path": str(path),
            "integrity": conn.execute("PRAGMA integrity_check").fetchone()[0],
            "chunks": int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
            "embeddings": int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]),
            "distinct_source_message_ids": len(chunk_message_ids),
            "missing_source_message_ids": len(source_message_ids - chunk_message_ids),
            "distinct_conversations": len(chunk_conversations),
            "missing_conversations": len(source_conversations - chunk_conversations),
            "chunk_kinds": dict(conn.execute("SELECT chunk_kind,COUNT(*) FROM chunks GROUP BY chunk_kind")),
            "roles": dict(conn.execute("SELECT role,COUNT(*) FROM chunks GROUP BY role")),
            "build": {
                "source_db": build[0],
                "vector_db": build[1],
                "message_count": int(build[2]),
                "chunk_count": int(build[3]),
                "token_estimate": int(build[4]),
                "small_cost_usd": float(build[5]),
                "large_cost_usd": float(build[6]),
                "preserve_source_messages": bool(build[7]),
            } if build else {},
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grok会話DBの整合性・件数・FTSを検証する")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--vector-db", default=str(DEFAULT_VECTOR_DB_PATH))
    parser.add_argument("--fts-query", default="grok")
    args = parser.parse_args()
    result = verify_database(args.db, fts_query=args.fts_query)
    result["vector_preparation"] = verify_vector_database(
        args.vector_db, source_db_path=args.db
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
