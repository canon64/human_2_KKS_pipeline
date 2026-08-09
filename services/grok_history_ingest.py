"""Grok会話エクスポート(ZIP)を差分だけ取り込み、ベクター化して索引まで更新する。

背景:
    Grokのエクスポートは毎回「今までの全部」が入ったZIPになる。
    素直に流すと同じ会話を何度も処理することになるので、各段で差分だけを扱う。

各段の差分の担保:
    1. import      import_grok.py が archive_sha256 で処理済みZIPをskip。
                   テーブルは ON CONFLICT DO UPDATE なので、再投入しても行は増えない。
    2. chunk       ここだけ既製品が使えない。vector_index.py は replace=True だと全消し、
                   --keep-existing だと素のINSERTで全件重複する。どちらも不可なので、
                   「既にchunksに居る message_id を除いてから追記する」処理を自前で持つ。
    3. embed       ollama_vectors.embed_pending_chunks が embeddings に未登録のchunkだけ拾う。
                   一番重い処理がここなので、差分が効くのが大きい。
    4. build-index FAISS は全再構築。頻度が低い前提なので条件分岐は入れない。

Ollama が要るのは 3 だけ。1,2,4 は無くても動く。
"""

from __future__ import annotations

import os
import subprocess
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_ROOT = Path(__file__).resolve().parents[1]
GROK_HISTORY_ROOT = _ROOT / "grok_history"
INCOMING_DIR = GROK_HISTORY_ROOT / "incoming"
DATA_DIR = GROK_HISTORY_ROOT / "runtime" / "data" / "grok_export_index"

RAW_DB = DATA_DIR / "grok_export_2026-07-11.sqlite"
VECTOR_DB = DATA_DIR / "grok_vectors_2026-07-11.sqlite"
INDEX_PATH = DATA_DIR / "vector_indexes" / "local_ollama_bge_m3_latest.faiss"
META_PATH = DATA_DIR / "vector_indexes" / "local_ollama_bge_m3_latest.meta.json"


@dataclass
class IngestResult:
    ok: bool = True
    steps: list[str] = field(default_factory=list)
    error: str = ""
    imported_zips: int = 0
    new_chunks: int = 0
    embedded: int = 0

    def log(self, message: str) -> None:
        self.steps.append(message)


def _python() -> Optional[Path]:
    """faissが使えるpythonを、サーバ起動と同じ規則で解決する。"""
    from services import grok_history_server

    return grok_history_server.resolve_python()


def _run(python_exe: Path, args: list[str], on_output: Callable[[str], None]) -> int:
    env = dict(os.environ)
    env["API_SCRIPTS_ROOT"] = str(GROK_HISTORY_ROOT)
    env["PYTHONPATH"] = str(GROK_HISTORY_ROOT)
    env["PYTHONUTF8"] = "1"

    process = subprocess.Popen(
        [str(python_exe), *args],
        cwd=str(GROK_HISTORY_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            on_output(line)
    return process.wait()


def find_zips() -> list[Path]:
    if not INCOMING_DIR.is_dir():
        return []
    return sorted(p for p in INCOMING_DIR.glob("*.zip") if p.is_file())


def append_new_chunks_only() -> int:
    """まだchunksに無い message_id 分だけをチャンク化して追記する。

    vector_index.build_chunks は「全メッセージ→全チャンク」しか作れないので、
    一旦全部作ってから、既にchunksに存在する message_id を落として書き込む。
    chunks には UNIQUE 制約が無いため、この除外を省くと素直に重複する。
    """
    sys.path.insert(0, str(GROK_HISTORY_ROOT))
    from scripts.chatgpt_export_browser.vector_index import build_chunks, load_messages

    messages = load_messages(RAW_DB)
    chunks = build_chunks(messages, preserve_source_messages=True)

    conn = sqlite3.connect(str(VECTOR_DB))
    try:
        known = {row[0] for row in conn.execute("SELECT DISTINCT message_id FROM chunks")}
        fresh = [c for c in chunks if c.message_id not in known]
        if not fresh:
            return 0
        with conn:
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
                        c.conversation_id, c.message_id, c.node_id, c.title, c.role,
                        c.create_iso, c.source_file, c.chunk_index, c.chunk_kind,
                        c.language, c.token_estimate, c.text, c.embedding_input,
                    )
                    for c in fresh
                ],
            )
        return len(fresh)
    finally:
        conn.close()


def pending_embeddings() -> int:
    if not VECTOR_DB.is_file():
        return 0
    conn = sqlite3.connect(str(VECTOR_DB))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM chunks c "
            "LEFT JOIN embeddings e ON e.chunk_id = c.id WHERE e.chunk_id IS NULL"
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def ingest(on_output: Optional[Callable[[str], None]] = None) -> IngestResult:
    emit = on_output or (lambda line: None)
    result = IngestResult()

    python_exe = _python()
    if python_exe is None:
        result.ok = False
        result.error = "no python with faiss (set H2K_GROK_HISTORY_PYTHON)"
        return result

    # 埋め込みにOllamaが要る。GUIを介さずバッチから叩かれることもあるので、
    # ここでも設定に従って起動を試みる(既に動いていれば何もしない)。
    from services import grok_history_server

    settings = grok_history_server.load_settings()
    ollama_reason = grok_history_server.start_ollama(settings=settings)
    if ollama_reason:
        emit(f"[warn] ollama: {ollama_reason}")

    # 1) ZIP取り込み（無ければ飛ばす。DBだけ更新したい場合もあるため）
    zips = find_zips()
    for zip_path in zips:
        emit(f"[1/4] import {zip_path.name}")
        code = _run(
            python_exe,
            ["-m", "scripts.grok_export_browser.import_grok", str(zip_path), "--db", str(RAW_DB)],
            emit,
        )
        if code != 0:
            result.ok = False
            result.error = f"import failed: {zip_path.name} (exit {code})"
            return result
        result.imported_zips += 1
    if not zips:
        emit(f"[1/4] import skipped (no zip in {INCOMING_DIR})")
    result.log(f"imported zips: {result.imported_zips}")

    # 2) 新規メッセージ分だけチャンク化
    emit("[2/4] chunk (new messages only)")
    try:
        result.new_chunks = append_new_chunks_only()
    except Exception as exc:  # noqa: BLE001 - 失敗理由をそのまま見せたい
        result.ok = False
        result.error = f"chunk failed: {exc}"
        return result
    emit(f"      new chunks: {result.new_chunks}")
    result.log(f"new chunks: {result.new_chunks}")

    # 3) 未埋め込みだけ埋め込む（Ollamaが要るのはここだけ）
    pending = pending_embeddings()
    emit(f"[3/4] embed pending={pending}")
    if pending > 0:
        code = _run(
            python_exe,
            ["-m", "scripts.grok_export_browser.ollama_vectors", "embed",
             "--vector-db", str(VECTOR_DB),
             "--endpoint", str(settings.get("grok_history_ollama_endpoint", ""))],
            emit,
        )
        if code != 0:
            result.ok = False
            result.error = f"embed failed (exit {code}). Ollama is running?"
            return result
        result.embedded = pending - pending_embeddings()
    result.log(f"embedded: {result.embedded}")

    # 4) 索引再構築。新規が1件も無ければ作り直す意味がない。
    if result.new_chunks == 0 and result.embedded == 0:
        emit("[4/4] build-index skipped (nothing new)")
        result.log("index: skipped")
        return result

    emit("[4/4] build-index")
    code = _run(
        python_exe,
        ["-m", "scripts.grok_export_browser.ollama_vectors", "build-index",
         "--vector-db", str(VECTOR_DB), "--index", str(INDEX_PATH), "--meta", str(META_PATH)],
        emit,
    )
    if code != 0:
        result.ok = False
        result.error = f"build-index failed (exit {code})"
        return result
    result.log("index: rebuilt")
    return result


def main() -> int:
    sys.path.insert(0, str(_ROOT))
    result = ingest(on_output=lambda line: print(line, flush=True))
    print("-" * 50)
    for step in result.steps:
        print("  " + step)
    if not result.ok:
        print("NG: " + result.error)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
