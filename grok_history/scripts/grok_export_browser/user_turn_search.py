from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence
import argparse
import json
import sqlite3

import faiss
import numpy as np

from scripts.grok_export_browser.ollama_vectors import (
    DB_MODEL,
    DEFAULT_ENDPOINT,
    DEFAULT_INDEX_PATH,
    DEFAULT_RAW_DB,
    DEFAULT_VECTOR_DB,
    EXPECTED_DIMENSIONS,
    OLLAMA_MODEL,
    SOURCE,
    normalize_vector,
    ollama_embed,
)


QueryEmbedder = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class AssistantReply:
    message_id: int
    node_id: str
    create_iso: str
    text: str


@dataclass(frozen=True)
class UserTurnHit:
    rank: int
    score: float
    matched_chunk_id: int
    matched_chunk_text: str
    message_id: int
    node_id: str
    conversation_id: str
    conversation_title: str
    create_iso: str
    user_text: str
    assistant_replies: tuple[AssistantReply, ...]

    @property
    def assistant_text(self) -> str:
        return "\n\n".join(reply.text for reply in self.assistant_replies)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["assistant_text"] = self.assistant_text
        return result


def _readonly_connection(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=120)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=120000")
    return conn


def query_embedding_input(query: str) -> str:
    """Return only the user's search text; never add role, title, IDs, or labels."""
    return query.strip()


class UserTurnSearchService:
    """Exact cosine search over Grok user-message chunks with direct replies attached."""

    def __init__(
        self,
        *,
        vector_db: str | Path = DEFAULT_VECTOR_DB,
        raw_db: str | Path = DEFAULT_RAW_DB,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        model: str = DB_MODEL,
        source: str = SOURCE,
        dimensions: int = EXPECTED_DIMENSIONS,
        endpoint: str = DEFAULT_ENDPOINT,
        ollama_model: str = OLLAMA_MODEL,
        embedder: QueryEmbedder | None = None,
    ) -> None:
        self.vector_db = Path(vector_db)
        self.raw_db = Path(raw_db)
        self.index_path = Path(index_path)
        self.model = model
        self.source = source
        self.dimensions = dimensions
        self.endpoint = endpoint
        self.ollama_model = ollama_model
        self._embedder = embedder or (
            lambda texts: ollama_embed(
                texts,
                endpoint=self.endpoint,
                ollama_model=self.ollama_model,
            )
        )
        self._chunk_rows: dict[int, dict[str, object]] = {}
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        self._index = faiss.read_index(str(self.index_path))
        if self._index.d != self.dimensions:
            raise RuntimeError(
                f"FAISS dimension mismatch: expected={self.dimensions} actual={self._index.d}"
            )
        self._load_user_chunks()

    @property
    def user_chunk_count(self) -> int:
        return len(self._chunk_rows)

    @property
    def unique_user_message_count(self) -> int:
        return len({int(row["message_id"]) for row in self._chunk_rows.values()})

    def _load_user_chunks(self) -> None:
        with _readonly_connection(self.vector_db) as conn:
            rows = conn.execute(
                """
                SELECT c.id AS chunk_id,c.message_id,c.conversation_id,c.text,
                       COALESCE(c.create_iso,'') AS create_iso
                FROM chunks c
                JOIN embeddings e ON e.chunk_id=c.id
                WHERE c.role='user'
                  AND c.chunk_kind IN ('text','code')
                  AND trim(c.text) != ''
                  AND e.model=? AND e.source=? AND e.dimensions=?
                  AND e.normalized=1 AND length(e.embedding)=?
                ORDER BY c.id
                """,
                (self.model, self.source, self.dimensions, self.dimensions * 4),
            ).fetchall()
        if not rows:
            raise RuntimeError(
                f"No user embeddings found: model={self.model} source={self.source}"
            )
        self._chunk_rows = {
            int(row["chunk_id"]): {
                "message_id": int(row["message_id"]),
                "conversation_id": str(row["conversation_id"]),
                "text": str(row["text"]),
                # 期間フィルタ用。日付は本来 _hydrate_hits で messages から取るが、
                # それだと top_k を絞った後になり、期間で弾くと候補が枯渇する。
                # 検索前に絞れるようチャンク側の日付をここで持っておく。
                "create_iso": str(row["create_iso"] or ""),
            }
            for row in rows
        }

    def stats(self) -> dict[str, int]:
        message_ids = {int(row["message_id"]) for row in self._chunk_rows.values()}
        linked = 0
        if message_ids:
            with _readonly_connection(self.raw_db) as conn:
                users = conn.execute(
                    "SELECT id,conversation_id,node_id FROM messages WHERE role='user'"
                ).fetchall()
                reply_parents = {
                    (str(row["conversation_id"]), str(row["parent_id"]))
                    for row in conn.execute(
                        """
                        SELECT conversation_id,parent_id FROM messages
                        WHERE role='assistant' AND parent_id IS NOT NULL
                        """
                    )
                }
                linked = sum(
                    1
                    for row in users
                    if int(row["id"]) in message_ids
                    and (str(row["conversation_id"]), str(row["node_id"])) in reply_parents
                )
        return {
            "user_chunks": self.user_chunk_count,
            "unique_user_messages": len(message_ids),
            "user_messages_with_direct_assistant": linked,
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        date_from: str = "",
        date_to: str = "",
    ) -> list[UserTurnHit]:
        """date_from/date_to は 'YYYY-MM-DD' 形式。空文字なら制限しない。

        期間の判定は create_iso の先頭10文字(日付部分)の文字列比較で行う。
        ISO8601 は桁が揃っているので辞書順比較がそのまま日付順になる。
        date_to は「その日を含む」ため <= で比較する。
        """
        input_text = query_embedding_input(query)
        if not input_text:
            raise ValueError("検索文を入力してください。")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        embedded = self._embedder([input_text])
        if len(embedded) != 1:
            raise RuntimeError(
                f"Query embedding count mismatch: expected=1 actual={len(embedded)}"
            )
        query_vector = normalize_vector(embedded[0], self.dimensions).reshape(1, -1)
        # The persisted index contains user and assistant chunks. Searching the full
        # exact IndexFlatIP result list and then retaining only known user chunk IDs
        # gives the same order as a user-only exact index without a slow BLOB reload.
        scores, ids = self._index.search(query_vector, int(self._index.ntotal))

        selected: list[tuple[float, int, dict[str, object]]] = []
        seen_message_ids: set[int] = set()
        for score, chunk_id in zip(scores[0], ids[0]):
            if int(chunk_id) < 0:
                continue
            row = self._chunk_rows.get(int(chunk_id))
            if row is None:
                continue
            message_id = int(row["message_id"])
            if message_id in seen_message_ids:
                continue
            if date_from or date_to:
                day = str(row.get("create_iso") or "")[:10]
                if not day:
                    continue
                if date_from and day < date_from:
                    continue
                if date_to and day > date_to:
                    continue
            seen_message_ids.add(message_id)
            selected.append((float(score), int(chunk_id), row))
            if len(selected) >= top_k:
                break
        return self._hydrate_hits(selected)

    def _hydrate_hits(
        self,
        selected: Sequence[tuple[float, int, dict[str, object]]],
    ) -> list[UserTurnHit]:
        hits: list[UserTurnHit] = []
        with _readonly_connection(self.raw_db) as conn:
            for rank, (score, chunk_id, chunk) in enumerate(selected, start=1):
                user = conn.execute(
                    """
                    SELECT m.id,m.node_id,m.conversation_id,m.create_iso,m.text,
                           COALESCE(c.title,'') AS conversation_title
                    FROM messages m
                    LEFT JOIN conversations c ON c.id=m.conversation_id
                    WHERE m.id=? AND m.role='user'
                    """,
                    (int(chunk["message_id"]),),
                ).fetchone()
                if user is None:
                    continue
                reply_rows = conn.execute(
                    """
                    SELECT id,node_id,COALESCE(create_iso,'') AS create_iso,COALESCE(text,'') AS text
                    FROM messages
                    WHERE conversation_id=? AND parent_id=? AND role='assistant'
                    ORDER BY create_time,id
                    """,
                    (str(user["conversation_id"]), str(user["node_id"])),
                ).fetchall()
                replies = tuple(
                    AssistantReply(
                        message_id=int(reply["id"]),
                        node_id=str(reply["node_id"]),
                        create_iso=str(reply["create_iso"]),
                        text=str(reply["text"]),
                    )
                    for reply in reply_rows
                )
                hits.append(
                    UserTurnHit(
                        rank=rank,
                        score=score,
                        matched_chunk_id=chunk_id,
                        matched_chunk_text=str(chunk["text"]),
                        message_id=int(user["id"]),
                        node_id=str(user["node_id"]),
                        conversation_id=str(user["conversation_id"]),
                        conversation_title=str(user["conversation_title"]),
                        create_iso=str(user["create_iso"] or ""),
                        user_text=str(user["text"] or ""),
                        assistant_replies=replies,
                    )
                )
        return hits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search Grok user turns and show direct replies")
    parser.add_argument("--vector-db", type=Path, default=DEFAULT_VECTOR_DB)
    parser.add_argument("--raw-db", type=Path, default=DEFAULT_RAW_DB)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--search", default="")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    service = UserTurnSearchService(
        vector_db=args.vector_db,
        raw_db=args.raw_db,
        index_path=args.index,
    )
    if args.stats:
        print(json.dumps(service.stats(), ensure_ascii=False, indent=2))
    if args.search:
        print(
            json.dumps(
                [hit.to_dict() for hit in service.search(args.search, top_k=args.top_k)],
                ensure_ascii=False,
                indent=2,
            )
        )
    if not args.stats and not args.search:
        print(json.dumps(service.stats(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
