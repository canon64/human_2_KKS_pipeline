from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request

import faiss
import numpy as np

from scripts.j_paths import data_path


DEFAULT_VECTOR_DB = data_path("grok_export_index", "grok_vectors_2026-07-11.sqlite")
DEFAULT_RAW_DB = data_path("grok_export_index", "grok_export_2026-07-11.sqlite")
DEFAULT_INDEX_DIR = data_path("grok_export_index", "vector_indexes")
DEFAULT_INDEX_PATH = DEFAULT_INDEX_DIR / "local_ollama_bge_m3_latest.faiss"
DEFAULT_META_PATH = DEFAULT_INDEX_DIR / "local_ollama_bge_m3_latest.meta.json"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
OLLAMA_MODEL = "bge-m3:latest"
DB_MODEL = "ollama:bge-m3:latest"
SOURCE = "ollama"
EXPECTED_DIMENSIONS = 1024
TARGET_WHERE = "c.role IN ('user','assistant') AND c.chunk_kind IN ('text','code') AND trim(c.text) != ''"

Embedder = Callable[[list[str]], list[list[float]]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_vector_db(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"Grok vector DB not found: {db_path}")
    if readonly:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=120)
    else:
        conn = sqlite3.connect(db_path, timeout=120)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_embedding_schema(conn: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(embeddings)")}
    if not columns:
        raise RuntimeError("embeddings table not found")
    if "source" not in columns:
        conn.execute("ALTER TABLE embeddings ADD COLUMN source TEXT NOT NULL DEFAULT ''")
    if "input_sha256" not in columns:
        conn.execute("ALTER TABLE embeddings ADD COLUMN input_sha256 TEXT NOT NULL DEFAULT ''")
    if "normalized" not in columns:
        conn.execute("ALTER TABLE embeddings ADD COLUMN normalized INTEGER NOT NULL DEFAULT 0")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model_source ON embeddings(model,source)")
    conn.commit()


def _valid_join(model: str, source: str, dimensions: int) -> tuple[str, tuple[object, ...]]:
    join = (
        "e.chunk_id=c.id AND e.model=? AND e.source=? AND e.dimensions=? "
        "AND length(e.embedding)=? AND e.normalized=1"
    )
    return join, (model, source, dimensions, dimensions * 4)


def target_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM chunks c WHERE {TARGET_WHERE}").fetchone()[0])


def pending_count(
    conn: sqlite3.Connection,
    model: str = DB_MODEL,
    *,
    source: str = SOURCE,
    dimensions: int = EXPECTED_DIMENSIONS,
) -> int:
    join, params = _valid_join(model, source, dimensions)
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM chunks c LEFT JOIN embeddings e ON {join} "
            f"WHERE {TARGET_WHERE} AND e.chunk_id IS NULL",
            params,
        ).fetchone()[0]
    )


def embedded_count(conn: sqlite3.Connection, model: str = DB_MODEL) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM embeddings WHERE model=?", (model,)).fetchone()[0])


def _next_batch(
    conn: sqlite3.Connection,
    model: str,
    *,
    source: str,
    dimensions: int,
    batch_size: int,
    max_batch_chars: int,
) -> list[sqlite3.Row]:
    join, params = _valid_join(model, source, dimensions)
    rows = conn.execute(
        f"SELECT c.id,c.text,c.role,c.message_id,c.conversation_id FROM chunks c "
        f"LEFT JOIN embeddings e ON {join} "
        f"WHERE {TARGET_WHERE} AND e.chunk_id IS NULL ORDER BY c.id LIMIT ?",
        (*params, batch_size),
    ).fetchall()
    selected: list[sqlite3.Row] = []
    total_chars = 0
    for row in rows:
        text = str(row["text"] or "").strip()
        if selected and total_chars + len(text) > max_batch_chars:
            break
        selected.append(row)
        total_chars += len(text)
    return selected


def body_only_input(row: sqlite3.Row) -> str:
    """Return only the stored message-body chunk; never add metadata or prefixes."""
    return str(row["text"] or "").strip()


def ollama_embed(
    texts: list[str],
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    ollama_model: str = OLLAMA_MODEL,
    timeout: float = 600.0,
    nan_retry_delays: tuple[float, ...] = (2.0, 5.0, 10.0),
) -> list[list[float]]:
    if not texts:
        return []
    payload = json.dumps({"model": ollama_model, "input": texts}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        if exc.code == 500 and "NaN" in detail and len(texts) > 1:
            midpoint = len(texts) // 2
            print(
                f"ollama_batch_retry_split inputs={len(texts)} left={midpoint} "
                f"right={len(texts) - midpoint} reason=nan",
                flush=True,
            )
            return ollama_embed(
                texts[:midpoint],
                endpoint=endpoint,
                ollama_model=ollama_model,
                timeout=timeout,
                nan_retry_delays=nan_retry_delays,
            ) + ollama_embed(
                texts[midpoint:],
                endpoint=endpoint,
                ollama_model=ollama_model,
                timeout=timeout,
                nan_retry_delays=nan_retry_delays,
            )
        if exc.code == 500 and "NaN" in detail and nan_retry_delays:
            delay = nan_retry_delays[0]
            print(
                f"ollama_single_retry inputs={len(texts)} delay={delay:g}s reason=nan",
                flush=True,
            )
            if delay > 0:
                time.sleep(delay)
            return ollama_embed(
                texts,
                endpoint=endpoint,
                ollama_model=ollama_model,
                timeout=timeout,
                nan_retry_delays=nan_retry_delays[1:],
            )
        raise RuntimeError(f"Ollama /api/embed failed: HTTP {exc.code}: {detail}") from exc
    vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise RuntimeError(
            f"Ollama embedding count mismatch: inputs={len(texts)} outputs={len(vectors or [])}"
        )
    return vectors


def normalize_vector(values: Iterable[float], expected_dimensions: int) -> np.ndarray:
    vector = np.asarray(list(values), dtype=np.float32)
    if vector.ndim != 1 or vector.size != expected_dimensions:
        raise RuntimeError(
            f"embedding dimension mismatch: expected={expected_dimensions} actual={vector.size}"
        )
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("embedding contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise RuntimeError("embedding has zero norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def embed_resilient(texts: list[str], embedder: Embedder) -> list[list[float]]:
    """Retry an Ollama-failing batch as smaller exact-body batches without altering text."""
    try:
        return embedder(texts)
    except RuntimeError as exc:
        if len(texts) == 1:
            text = texts[0]
            if "NaN" not in str(exc) or len(text) <= 1:
                raise
            midpoint = len(text) // 2
            left_text = text[:midpoint]
            right_text = text[midpoint:]
            print(
                f"embedding_body_split chars={len(text)} left={len(left_text)} "
                f"right={len(right_text)} reason=nan",
                flush=True,
            )
            left = np.asarray(embed_resilient([left_text], embedder)[0], dtype=np.float32)
            right = np.asarray(embed_resilient([right_text], embedder)[0], dtype=np.float32)
            if left.shape != right.shape:
                raise RuntimeError(
                    f"split embedding shape mismatch: left={left.shape} right={right.shape}"
                )
            combined = (left * len(left_text) + right * len(right_text)) / len(text)
            return [combined.astype(np.float32).tolist()]
        midpoint = len(texts) // 2
        print(
            f"embedding_batch_split failed_size={len(texts)} "
            f"left={midpoint} right={len(texts) - midpoint}",
            flush=True,
        )
        return embed_resilient(texts[:midpoint], embedder) + embed_resilient(
            texts[midpoint:], embedder
        )


def embed_pending_chunks(
    vector_db: str | Path = DEFAULT_VECTOR_DB,
    *,
    model: str = DB_MODEL,
    source: str = SOURCE,
    endpoint: str = DEFAULT_ENDPOINT,
    ollama_model: str = OLLAMA_MODEL,
    expected_dimensions: int = EXPECTED_DIMENSIONS,
    batch_size: int = 32,
    max_batch_chars: int = 100_000,
    timeout: float = 600.0,
    limit: int | None = None,
    embedder: Embedder | None = None,
) -> dict[str, object]:
    conn = connect_vector_db(vector_db)
    ensure_embedding_schema(conn)
    total_target = target_count(conn)
    started_pending = pending_count(conn, model, source=source, dimensions=expected_dimensions)
    embedded = 0
    requests = 0
    dimensions = 0
    if embedder is None:
        embedder = lambda texts: ollama_embed(
            texts, endpoint=endpoint, ollama_model=ollama_model, timeout=timeout
        )
    try:
        while limit is None or embedded < limit:
            effective_size = batch_size if limit is None else min(batch_size, limit - embedded)
            if effective_size <= 0:
                break
            rows = _next_batch(
                conn,
                model,
                source=source,
                dimensions=expected_dimensions,
                batch_size=effective_size,
                max_batch_chars=max_batch_chars,
            )
            if not rows:
                break
            inputs = [body_only_input(row) for row in rows]
            if any(not text for text in inputs):
                raise RuntimeError("empty body reached embedding request")
            vectors = embed_resilient(inputs, embedder)
            if len(vectors) != len(rows):
                raise RuntimeError(
                    f"embedding count mismatch: rows={len(rows)} vectors={len(vectors)}"
                )
            created_at = utc_now()
            records = []
            for row, text, values in zip(rows, inputs, vectors):
                vector = normalize_vector(values, expected_dimensions)
                dimensions = int(vector.size)
                records.append(
                    (
                        int(row["id"]), model, dimensions, vector.tobytes(), created_at,
                        source, hashlib.sha256(text.encode("utf-8")).hexdigest(), 1,
                    )
                )
            with conn:
                conn.executemany(
                    """
                    INSERT INTO embeddings(
                        chunk_id,model,dimensions,embedding,created_at,source,input_sha256,normalized
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id,model) DO UPDATE SET
                        dimensions=excluded.dimensions,embedding=excluded.embedding,
                        created_at=excluded.created_at,source=excluded.source,
                        input_sha256=excluded.input_sha256,normalized=excluded.normalized
                    """,
                    records,
                )
            embedded += len(records)
            requests += 1
            remaining = pending_count(conn, model, source=source, dimensions=expected_dimensions)
            print(
                f"embedded_batch={len(records)} embedded_this_run={embedded} "
                f"embedded_total={embedded_count(conn, model)} pending={remaining} "
                f"dimensions={dimensions} source={source} model={model}",
                flush=True,
            )
    finally:
        final_embedded = embedded_count(conn, model)
        final_pending = pending_count(conn, model, source=source, dimensions=expected_dimensions)
        conn.close()
    return {
        "vector_db": str(Path(vector_db)),
        "source": source,
        "model": model,
        "ollama_model": ollama_model,
        "target_chunks": total_target,
        "started_pending": started_pending,
        "embedded_this_run": embedded,
        "embedded_total": final_embedded,
        "pending": final_pending,
        "requests": requests,
        "dimensions": dimensions or expected_dimensions,
    }


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json.tmp", delete=False, encoding="utf-8", dir=path.parent
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _embedding_rows(
    conn: sqlite3.Connection,
    model: str,
    source: str,
    dimensions: int,
) -> list[sqlite3.Row]:
    return conn.execute(
        f"""
        SELECT e.chunk_id,e.embedding,c.role
        FROM embeddings e JOIN chunks c ON c.id=e.chunk_id
        WHERE e.model=? AND e.source=? AND e.dimensions=? AND length(e.embedding)=?
          AND e.normalized=1 AND {TARGET_WHERE}
        ORDER BY e.chunk_id
        """,
        (model, source, dimensions, dimensions * 4),
    ).fetchall()


def build_faiss_index(
    vector_db: str | Path = DEFAULT_VECTOR_DB,
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
    model: str = DB_MODEL,
    source: str = SOURCE,
    dimensions: int = EXPECTED_DIMENSIONS,
) -> dict[str, object]:
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_vector_db(vector_db, readonly=True)
    try:
        total_target = target_count(conn)
        rows = _embedding_rows(conn, model, source, dimensions)
        if len(rows) != total_target:
            raise RuntimeError(
                f"cannot build incomplete index: target={total_target} embeddings={len(rows)}"
            )
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimensions))
        role_counts: dict[str, int] = {}
        for start in range(0, len(rows), 1024):
            batch = rows[start : start + 1024]
            ids = np.asarray([int(row["chunk_id"]) for row in batch], dtype=np.int64)
            vectors = np.vstack(
                [np.frombuffer(row["embedding"], dtype=np.float32) for row in batch]
            ).astype(np.float32, copy=False)
            if vectors.shape != (len(batch), dimensions):
                raise RuntimeError(f"invalid vector matrix shape: {vectors.shape}")
            faiss.normalize_L2(vectors)
            index.add_with_ids(np.ascontiguousarray(vectors), ids)
            for row in batch:
                role = str(row["role"])
                role_counts[role] = role_counts.get(role, 0) + 1
            print(f"index_progress added={index.ntotal} total={len(rows)}", flush=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=index_path.name + ".", suffix=".tmp", dir=index_path.parent
        )
        os.close(fd)
        temp_index = Path(temp_name)
        try:
            faiss.write_index(index, str(temp_index))
            os.replace(temp_index, index_path)
        finally:
            if temp_index.exists():
                temp_index.unlink()
        chunk_ids = [int(row["chunk_id"]) for row in rows]
        meta: dict[str, object] = {
            "schema": "grok_ollama_faiss/v1",
            "dataset": "grok",
            "source": source,
            "model": model,
            "ollama_model": OLLAMA_MODEL,
            "dimensions": dimensions,
            "index_type": "IndexIDMap2(IndexFlatIP)",
            "metric": "cosine",
            "normalized": True,
            "id_column": "chunks.id",
            "vector_db": str(Path(vector_db)),
            "index_path": str(index_path),
            "target_chunks": total_target,
            "indexed_rows": int(index.ntotal),
            "role_counts": role_counts,
            "min_chunk_id": min(chunk_ids) if chunk_ids else None,
            "max_chunk_id": max(chunk_ids) if chunk_ids else None,
            "created_at": utc_now(),
        }
        write_json_atomic(meta_path, meta)
        print(
            f"index_done path={index_path} meta={meta_path} ntotal={index.ntotal}",
            flush=True,
        )
        return meta
    finally:
        conn.close()


def search_index(
    query: str,
    *,
    vector_db: str | Path = DEFAULT_VECTOR_DB,
    raw_db: str | Path | None = DEFAULT_RAW_DB,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    top_k: int = 10,
    endpoint: str = DEFAULT_ENDPOINT,
    ollama_model: str = OLLAMA_MODEL,
    dimensions: int = EXPECTED_DIMENSIONS,
    embedder: Embedder | None = None,
) -> list[dict[str, object]]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if embedder is None:
        embedder = lambda texts: ollama_embed(texts, endpoint=endpoint, ollama_model=ollama_model)
    query_vector = normalize_vector(embedder([query])[0], dimensions).reshape(1, -1)
    index = faiss.read_index(str(index_path))
    scores, ids = index.search(query_vector, min(max(top_k, 1), max(int(index.ntotal), 1)))
    conn = connect_vector_db(vector_db, readonly=True)
    raw_conn = sqlite3.connect(raw_db) if raw_db and Path(raw_db).exists() else None
    try:
        results: list[dict[str, object]] = []
        for score, chunk_id in zip(scores[0], ids[0]):
            if int(chunk_id) < 0:
                continue
            row = conn.execute(
                """
                SELECT id,conversation_id,message_id,node_id,role,chunk_index,chunk_kind,text
                FROM chunks WHERE id=?
                """,
                (int(chunk_id),),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"FAISS chunk id not found in DB: {chunk_id}")
            raw_matches = None
            if raw_conn is not None:
                raw = raw_conn.execute("SELECT text FROM messages WHERE id=?", (int(row["message_id"]),)).fetchone()
                normalized_raw = str(raw[0] if raw else "").replace("\r\n", "\n").replace("\r", "\n")
                raw_matches = str(row["text"]) in normalized_raw
            results.append(
                {
                    "score": float(score),
                    "chunk_id": int(row["id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "message_id": int(row["message_id"]),
                    "node_id": str(row["node_id"] or ""),
                    "role": str(row["role"]),
                    "chunk_index": int(row["chunk_index"]),
                    "chunk_kind": str(row["chunk_kind"]),
                    "text": str(row["text"]),
                    "raw_message_contains_chunk": raw_matches,
                }
            )
        return results
    finally:
        if raw_conn is not None:
            raw_conn.close()
        conn.close()


def verify_grok_vectors(
    vector_db: str | Path = DEFAULT_VECTOR_DB,
    *,
    raw_db: str | Path = DEFAULT_RAW_DB,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
    model: str = DB_MODEL,
    source: str = SOURCE,
    dimensions: int = EXPECTED_DIMENSIONS,
) -> dict[str, object]:
    conn = connect_vector_db(vector_db, readonly=True)
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        targets = target_count(conn)
        target_roles = dict(
            conn.execute(f"SELECT c.role,COUNT(*) FROM chunks c WHERE {TARGET_WHERE} GROUP BY c.role")
        )
        embedding_roles = dict(
            conn.execute(
                f"""
                SELECT c.role,COUNT(*) FROM embeddings e JOIN chunks c ON c.id=e.chunk_id
                WHERE e.model=? AND e.source=? AND {TARGET_WHERE} GROUP BY c.role
                """,
                (model, source),
            )
        )
        embedded = int(
            conn.execute("SELECT COUNT(*) FROM embeddings WHERE model=? AND source=?", (model, source)).fetchone()[0]
        )
        outside_target = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM embeddings e JOIN chunks c ON c.id=e.chunk_id
                WHERE e.model=? AND e.source=? AND NOT ({TARGET_WHERE})
                """,
                (model, source),
            ).fetchone()[0]
        )
        empty_inputs = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM embeddings e JOIN chunks c ON c.id=e.chunk_id
                WHERE e.model=? AND e.source=? AND trim(c.text)=''
                """,
                (model, source),
            ).fetchone()[0]
        )
        rows = conn.execute(
            """
            SELECT e.chunk_id,e.dimensions,length(e.embedding) AS bytes,e.embedding,
                   e.input_sha256,e.normalized,c.message_id,c.text
            FROM embeddings e JOIN chunks c ON c.id=e.chunk_id
            WHERE e.model=? AND e.source=? ORDER BY e.chunk_id
            """,
            (model, source),
        ).fetchall()
        dimension_values = sorted({int(row["dimensions"]) for row in rows})
        byte_lengths = sorted({int(row["bytes"]) for row in rows})
        hash_mismatches = 0
        norms: list[float] = []
        raw_messages: dict[int, str] = {}
        with sqlite3.connect(raw_db) as raw_conn:
            raw_messages = {
                int(message_id): str(text or "").replace("\r\n", "\n").replace("\r", "\n")
                for message_id, text in raw_conn.execute("SELECT id,text FROM messages")
            }
        raw_mismatches = 0
        for row in rows:
            text = str(row["text"] or "").strip()
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != str(row["input_sha256"]):
                hash_mismatches += 1
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            norms.append(float(np.linalg.norm(vector)))
            if str(row["text"]) not in raw_messages.get(int(row["message_id"]), ""):
                raw_mismatches += 1
        index = faiss.read_index(str(index_path))
        index_ids = [int(value) for value in faiss.vector_to_array(index.id_map)]
        embedding_ids = [int(row["chunk_id"]) for row in rows]
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return {
            "integrity": integrity,
            "vector_db": str(Path(vector_db)),
            "raw_db": str(Path(raw_db)),
            "source": source,
            "model": model,
            "target_chunks": targets,
            "embedded": embedded,
            "pending": targets - embedded,
            "target_role_counts": target_roles,
            "embedding_role_counts": embedding_roles,
            "outside_target_embeddings": outside_target,
            "empty_input_embeddings": empty_inputs,
            "dimensions": dimension_values,
            "vector_byte_lengths": byte_lengths,
            "normalized_flags": sorted({int(row["normalized"]) for row in rows}),
            "norm_min": min(norms) if norms else None,
            "norm_max": max(norms) if norms else None,
            "input_sha256_mismatches": hash_mismatches,
            "raw_message_mismatches": raw_mismatches,
            "faiss_class": type(index).__name__,
            "faiss_dimensions": int(index.d),
            "faiss_ntotal": int(index.ntotal),
            "faiss_ids_match_db": index_ids == embedding_ids,
            "meta": meta,
            "meta_matches": (
                int(meta.get("indexed_rows") or -1) == int(index.ntotal) == embedded
                and int(meta.get("dimensions") or -1) == dimensions == int(index.d)
                and str(meta.get("model")) == model
                and str(meta.get("source")) == source
            ),
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grok本文をlocal Ollamaで埋め込み・FAISS検索する")
    subparsers = parser.add_subparsers(dest="command", required=True)
    embed_parser = subparsers.add_parser("embed")
    embed_parser.add_argument("--vector-db", default=str(DEFAULT_VECTOR_DB))
    embed_parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    embed_parser.add_argument("--batch-size", type=int, default=32)
    embed_parser.add_argument("--max-batch-chars", type=int, default=100_000)
    embed_parser.add_argument("--timeout", type=float, default=600.0)
    embed_parser.add_argument("--limit", type=int, default=None)
    build_parser = subparsers.add_parser("build-index")
    build_parser.add_argument("--vector-db", default=str(DEFAULT_VECTOR_DB))
    build_parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    build_parser.add_argument("--meta", default=str(DEFAULT_META_PATH))
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--vector-db", default=str(DEFAULT_VECTOR_DB))
    search_parser.add_argument("--raw-db", default=str(DEFAULT_RAW_DB))
    search_parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--vector-db", default=str(DEFAULT_VECTOR_DB))
    verify_parser.add_argument("--raw-db", default=str(DEFAULT_RAW_DB))
    verify_parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    verify_parser.add_argument("--meta", default=str(DEFAULT_META_PATH))
    args = parser.parse_args()
    if args.command == "embed":
        result = embed_pending_chunks(
            args.vector_db,
            endpoint=args.endpoint,
            batch_size=args.batch_size,
            max_batch_chars=args.max_batch_chars,
            timeout=args.timeout,
            limit=args.limit,
        )
    elif args.command == "build-index":
        result = build_faiss_index(args.vector_db, index_path=args.index, meta_path=args.meta)
    elif args.command == "search":
        result = search_index(
            args.query,
            vector_db=args.vector_db,
            raw_db=args.raw_db,
            index_path=args.index,
            top_k=args.top_k,
            endpoint=args.endpoint,
        )
    else:
        result = verify_grok_vectors(
            args.vector_db,
            raw_db=args.raw_db,
            index_path=args.index,
            meta_path=args.meta,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
