from __future__ import annotations

import argparse
import ipaddress
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from scripts.grok_export_browser.ollama_vectors import (
    DEFAULT_ENDPOINT,
    DEFAULT_INDEX_PATH,
    DEFAULT_RAW_DB,
    DEFAULT_VECTOR_DB,
    OLLAMA_MODEL,
)
from scripts.grok_export_browser.user_turn_search import UserTurnSearchService


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8877
MAX_REQUEST_BYTES = 1024 * 1024
LOGGER = logging.getLogger("grok_history_response_api")


def _normalize_date(value: object) -> str:
    """'YYYY-MM-DD' だけ通す。空や不正な形式は「制限なし」として空文字を返す。"""
    token = str(value or "").strip()
    if not token:
        return ""
    if len(token) != 10 or token[4] != "-" or token[7] != "-":
        raise ValueError(f"date must be YYYY-MM-DD: {token}")
    if not (token[:4].isdigit() and token[5:7].isdigit() and token[8:].isdigit()):
        raise ValueError(f"date must be YYYY-MM-DD: {token}")
    return token


def _is_loopback(host: str) -> bool:
    token = str(host or "").strip().lower()
    if token == "localhost":
        return True
    try:
        return ipaddress.ip_address(token).is_loopback
    except ValueError:
        return False


class GrokHistoryApiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], service: Any) -> None:
        self.service = service
        super().__init__(server_address, GrokHistoryApiHandler)


class GrokHistoryApiHandler(BaseHTTPRequestHandler):
    server: GrokHistoryApiServer

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length is invalid") from exc
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body size is invalid")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request JSON root must be an object")
        return payload

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        service = self.server.service
        self._write_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "service": "grok-history-response-api",
                "stats": service.stats(),
                "vector_db": str(getattr(service, "vector_db", "")),
                "raw_db": str(getattr(service, "raw_db", "")),
                "index": str(getattr(service, "index_path", "")),
                "model": str(getattr(service, "model", "")),
                "source": str(getattr(service, "source", "")),
                "dimensions": int(getattr(service, "dimensions", 0) or 0),
            },
        )

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/search":
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        try:
            payload = self._read_json()
            query = str(payload.get("query") or "").strip()
            if not query:
                raise ValueError("query is required")
            top_k = int(payload.get("top_k", 1))
            if top_k < 1 or top_k > 100:
                raise ValueError("top_k must be between 1 and 100")
            date_from = _normalize_date(payload.get("date_from"))
            date_to = _normalize_date(payload.get("date_to"))
            hits = self.server.service.search(
                query, top_k=top_k, date_from=date_from, date_to=date_to
            )
            results = [hit.to_dict() for hit in hits]
            selected = results[0] if results else None
            if selected is None:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "no matching user message", "query": query},
                )
                return
            if not str(selected.get("assistant_text") or "").strip():
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {
                        "ok": False,
                        "error": "top user message has no direct assistant reply",
                        "query": query,
                        "selected": selected,
                    },
                )
                return
            LOGGER.info(
                "history_match conversation_id=%s user_message_id=%s score=%.6f assistant_count=%d",
                selected.get("conversation_id"),
                selected.get("message_id"),
                float(selected.get("score") or 0.0),
                len(selected.get("assistant_replies") or []),
            )
            self._write_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "query": query,
                    "top_k": top_k,
                    "count": len(results),
                    "selected": selected,
                    "results": results,
                },
            )
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:
            LOGGER.exception("search_failed")
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"history search failed: {exc}"},
            )

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("http client=%s %s", self.client_address[0], format % args)


def build_server(host: str, port: int, service: Any) -> GrokHistoryApiServer:
    if not _is_loopback(host):
        raise ValueError(f"host must be loopback: {host}")
    return GrokHistoryApiServer((host, int(port)), service)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Loopback API for Grok user-turn vector search and direct assistant replies"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--vector-db", type=Path, default=DEFAULT_VECTOR_DB)
    parser.add_argument("--raw-db", type=Path, default=DEFAULT_RAW_DB)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Ollama endpoint used to embed the query (e.g. http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL,
        help="Embedding model. Must match the model the index was built with.",
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--search", default="")
    parser.add_argument("--top-k", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    service = UserTurnSearchService(
        vector_db=args.vector_db,
        raw_db=args.raw_db,
        index_path=args.index,
        endpoint=args.endpoint,
        ollama_model=args.ollama_model,
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
    if args.serve:
        server = build_server(args.host, args.port, service)
        LOGGER.info(
            "api_start host=%s port=%d vector_db=%s raw_db=%s index=%s",
            args.host,
            args.port,
            args.vector_db,
            args.raw_db,
            args.index,
        )
        try:
            server.serve_forever(poll_interval=0.25)
        finally:
            server.server_close()
        return 0
    if not args.stats and not args.search:
        print(json.dumps(service.stats(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
