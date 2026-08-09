from __future__ import annotations

import ipaddress
import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit

from core.log_safety import sanitize_log_text

from .config import BridgeConfig


@dataclass(frozen=True)
class LlmRequestConfig:
    backend: str = "grok_browser"
    grok_history_enabled: bool = True
    grok_history_search_url: str = ""
    grok_history_top_k: int = 10
    grok_history_selection_mode: str = "best"
    grok_history_min_score: float = -1.0
    grok_history_timeout_seconds: float = 0.0
    grok_history_fallback_live: bool = False
    grok_history_required_match_mode: str = "any"
    grok_history_response_required_terms: str = ""
    grok_history_response_preferred_terms: str = ""
    grok_history_date_from: str = ""
    grok_history_date_to: str = ""
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = ""
    api_key: str = "lm-studio"
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_seconds: float = 120.0


# 中身を読みに行く拡張子。一覧をJSONやMarkdownで管理したい場合があるため。
_APPEND_FILE_SUFFIXES = (".txt", ".json", ".md", ".csv")


def resolve_append_source(value: str) -> str:
    """文言そのものか、ファイルのパスかを見分けて中身を返す。

    対応拡張子で終わり実在すればその中身（UTF-8）を返す。相対パスはプラグイン
    フォルダ起点で解決する。見つからなければ文字列そのものを文言として扱う。
    """
    raw = str(value or "").strip()
    # Windows の「パスのコピー」は引用符付きで貼られる。囲みを外して判定する。
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        raw = raw[1:-1].strip()
    if not raw or not raw.lower().endswith(_APPEND_FILE_SUFFIXES):
        return raw
    candidates = [raw]
    if not os.path.isabs(raw):
        from .config import runtime_base_dir

        candidates.append(os.path.join(runtime_base_dir(), raw))
    for path in candidates:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    return handle.read().strip()
        except Exception:
            continue
    return raw


def _keyword_rule_matches(rule: dict[str, Any], text: str) -> bool:
    pattern = str(rule.get("pattern") or "").strip()
    if not pattern:
        return False
    kind = str(rule.get("type") or "partial").strip().lower()
    if kind == "exact":
        return text.strip() == pattern
    if kind == "regex":
        try:
            return re.search(pattern, text) is not None
        except re.error:
            return False
    return pattern in text


def collect_keyword_appends(text: str, rules: list[dict[str, Any]] | None) -> list[str]:
    """発話に当たったルールの文言を、登録順に重複なく返す。"""
    body = str(text or "")
    found: list[str] = []
    for rule in rules or []:
        if not isinstance(rule, dict) or not rule.get("enabled", True):
            continue
        if not _keyword_rule_matches(rule, body):
            continue
        appended = resolve_append_source(rule.get("append", ""))
        if appended and appended not in found:
            found.append(appended)
    return found


def compose_llm_input(
    text: str,
    always_append_text: str,
    keyword_rules: list[dict[str, Any]] | None = None,
) -> str:
    """LLM/検索へ送る文にだけ、固定の追記とワード連動の追記を足す。

    読み上げ側には影響しない（元の発話をそのまま喋らせるため）。
    """
    base = str(text or "").rstrip()
    parts: list[str] = []
    fixed = str(always_append_text or "").strip()
    if fixed:
        parts.append(fixed)
    parts.extend(collect_keyword_appends(base, keyword_rules))
    if not parts:
        return str(text or "")
    suffix = "\n".join(parts)
    if not base:
        return suffix
    return f"{base} {suffix}"


def normalize_backend(value: str) -> str:
    token = str(value or "").strip().lower()
    aliases = {
        "": "grok_browser",
        "grok": "grok_browser",
        "browser": "grok_browser",
        "grok_browser": "grok_browser",
        "local": "local_openai",
        "local_llm": "local_openai",
        "local_openai": "local_openai",
        "openai_compatible": "local_openai",
        "lmstudio": "local_openai",
        "lm_studio": "local_openai",
        "ollama": "local_openai",
        "ollama_openai": "local_openai",
        "runpod": "runpod_openwebui",
        "run_pod": "runpod_openwebui",
        "runpod_openwebui": "runpod_openwebui",
        "openwebui": "runpod_openwebui",
        "open_webui": "runpod_openwebui",
    }
    if token not in aliases:
        raise ValueError(f"unsupported llm backend: {value}")
    return aliases[token]


# Pod ID だけ書かれた場合に展開するテンプレート。
# RunPod のプロキシは "<podId>-<port>.proxy.runpod.net" 形式で公開される。
_RUNPOD_PROXY_TEMPLATE = "https://{pod_id}-8080.proxy.runpod.net"
_RUNPOD_POD_ID_PATTERN = re.compile(r"^[a-z0-9]{8,}$", re.IGNORECASE)
_RUNPOD_API_SUFFIX = "/ollama/v1"

# RunPod のプロキシは Cloudflare 配下にあり、既定の `Python-urllib/x.y` という
# User-Agent は error code 1010 で遮断される（実測）。一般的な名乗りに差し替える。
_RUNPOD_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# (base_url, email) -> token。Pod を作り直すと base_url が変わるためキーに含める。
_runpod_token_cache: dict[tuple[str, str], str] = {}
_runpod_token_lock = threading.Lock()


_RUNPOD_CHAT_ID_PATTERN = re.compile(r"/c/([0-9a-f-]{16,})", re.IGNORECASE)


def split_runpod_target(value: str) -> tuple[str, str]:
    """base欄の入力を (API用URL, 会話ID) に分解する。

    ブラウザの会話URLをそのまま貼れるようにする。会話IDが取れた場合、
    その会話の続きとして送るので文脈が続く（Grok経路と同じ挙動になる）。
    会話URLでなければ会話IDは空になり、従来通り一問一答で送る。
    """
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        raise RuntimeError(
            "runpod_openwebui requires llm_base_url (Pod ID / URL / 会話URL)"
        )
    chat_id = ""
    matched = _RUNPOD_CHAT_ID_PATTERN.search(raw)
    if matched:
        chat_id = matched.group(1)
        raw = raw[: matched.start()]  # /c/<id> より前だけ残す

    raw = raw.rstrip("/")
    if "://" in raw:
        base = raw
    elif _RUNPOD_POD_ID_PATTERN.match(raw):
        base = _RUNPOD_PROXY_TEMPLATE.format(pod_id=raw)
    else:
        raise RuntimeError(
            f"runpod_openwebui: llm_base_url is neither a Pod ID nor a URL: {raw}"
        )
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if not base.endswith(_RUNPOD_API_SUFFIX):
        base = f"{base}{_RUNPOD_API_SUFFIX}"
    return base, chat_id


def normalize_runpod_base_url(value: str) -> str:
    """Pod ID もフルURLも受け取り、chat/completions の直前までのURLに正規化する。"""
    return split_runpod_target(value)[0]


def split_runpod_credentials(api_key: str) -> tuple[str, str] | None:
    """`email:password` ならその組を返す。トークン直指定なら None。"""
    raw = str(api_key or "").strip()
    if not raw or ":" not in raw:
        return None
    email, _, password = raw.partition(":")
    email = email.strip()
    password = password.strip()
    if not email or not password or "@" not in email:
        return None
    return email, password


def _runpod_webui_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith(_RUNPOD_API_SUFFIX):
        root = root[: -len(_RUNPOD_API_SUFFIX)]
    return root.rstrip("/")


def _runpod_signin(
    base_url: str,
    email: str,
    password: str,
    timeout: float,
    logger: logging.Logger,
) -> str:
    url = f"{_runpod_webui_root(base_url)}/api/v1/auths/signin"
    payload = json.dumps({"email": email, "password": password}, ensure_ascii=False)
    req = urllib.request.Request(url, data=payload.encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    req.add_header("User-Agent", _RUNPOD_USER_AGENT)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(
            f"runpod_openwebui signin HTTP {exc.code}: "
            f"{sanitize_log_text(detail, max_chars=300)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "runpod_openwebui signin failed (Pod が停止しているか URL が古い可能性があります): "
            f"{exc}"
        ) from exc

    try:
        data: Any = json.loads(body)
    except Exception as exc:
        raise RuntimeError("runpod_openwebui signin returned invalid JSON") from exc
    token = str(data.get("token") or "").strip() if isinstance(data, dict) else ""
    if not token:
        raise RuntimeError("runpod_openwebui signin response has no token")
    logger.info("runpod_signin_ok base_url=%s", base_url)
    return token


def _runpod_resolve_token(
    base_url: str,
    api_key: str,
    timeout: float,
    logger: logging.Logger,
    *,
    force_refresh: bool = False,
) -> str:
    credentials = split_runpod_credentials(api_key)
    if credentials is None:
        token = str(api_key or "").strip()
        if not token:
            raise RuntimeError(
                "runpod_openwebui requires llm_api_key "
                "(`email:password` もしくはトークン文字列)"
            )
        return token
    email, password = credentials
    cache_key = (base_url, email)
    with _runpod_token_lock:
        if not force_refresh:
            cached = _runpod_token_cache.get(cache_key)
            if cached:
                return cached
        token = _runpod_signin(base_url, email, password, timeout, logger)
        _runpod_token_cache[cache_key] = token
        return token


def _runpod_state_path() -> str:
    """会話IDの控え。プラグインフォルダ内に置く（絶対パスを埋め込まない）。"""
    from .config import runtime_base_dir

    return os.path.join(runtime_base_dir(), "runpod_chat_state.json")


def _runpod_load_saved_chat_id(base_url: str) -> str:
    try:
        with open(_runpod_state_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return str(data.get(base_url) or "")
    except Exception:
        return ""


def _runpod_save_chat_id(base_url: str, chat_id: str) -> None:
    path = _runpod_state_path()
    data: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            data = {str(k): str(v) for k, v in loaded.items()}
    except Exception:
        data = {}
    data[base_url] = chat_id
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 控えが残せなくても会話自体は成立するので落とさない


def _runpod_api(
    base_url: str,
    path: str,
    token: str,
    timeout: float,
    *,
    payload: object | None = None,
) -> Any:
    url = f"{_runpod_webui_root(base_url)}{path}"
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", _RUNPOD_USER_AGENT)
    if data:
        req.add_header("Content-Type", "application/json; charset=utf-8")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body) if body.strip() else {}


def _runpod_ensure_chat(
    base_url: str,
    chat_id: str,
    token: str,
    timeout: float,
    model: str,
    logger: logging.Logger,
) -> tuple[str, list[dict[str, Any]]]:
    """会話IDと、その会話のこれまでのやり取りを返す。無ければ新しく作る。"""
    if chat_id:
        try:
            data = _runpod_api(base_url, f"/api/v1/chats/{chat_id}", token, timeout)
            chat = data.get("chat") if isinstance(data, dict) else None
            messages = chat.get("messages") if isinstance(chat, dict) else None
            if isinstance(messages, list):
                return chat_id, [m for m in messages if isinstance(m, dict)]
        except Exception as exc:
            logger.warning(
                "runpod_chat_load_failed id=%s reason=%s",
                chat_id,
                sanitize_log_text(exc, max_chars=150),
            )
        return chat_id, []

    created = _runpod_api(
        base_url,
        "/api/v1/chats/new",
        token,
        timeout,
        payload={"chat": {"title": "Human2KKS", "models": [model], "messages": [], "history": {"messages": {}, "currentId": None}}},
    )
    new_id = str(created.get("id") or "") if isinstance(created, dict) else ""
    if new_id:
        logger.info("runpod_chat_created id=%s", new_id)
    return new_id, []


def _runpod_append_chat(
    base_url: str,
    chat_id: str,
    token: str,
    timeout: float,
    model: str,
    messages: list[dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """やり取りを会話スレッドへ書き戻す。ブラウザで開いたときに同じ内容が見える。"""
    if not chat_id:
        return
    history_messages: dict[str, Any] = {}
    previous_id: str | None = None
    current_id: str | None = None
    for index, message in enumerate(messages):
        message_id = str(message.get("id") or f"m{index}")
        history_messages[message_id] = {
            **message,
            "id": message_id,
            "parentId": previous_id,
            "childrenIds": [],
        }
        if previous_id is not None:
            history_messages[previous_id]["childrenIds"] = [message_id]
        previous_id = message_id
        current_id = message_id
    payload = {
        "chat": {
            "models": [model],
            "messages": messages,
            "history": {"messages": history_messages, "currentId": current_id},
        }
    }
    try:
        _runpod_api(base_url, f"/api/v1/chats/{chat_id}", token, timeout, payload=payload)
    except Exception as exc:
        logger.warning(
            "runpod_chat_save_failed id=%s reason=%s",
            chat_id,
            sanitize_log_text(exc, max_chars=150),
        )


def list_runpod_chats(
    base_url: str,
    api_key: str,
    *,
    timeout_seconds: float = 30.0,
    logger: logging.Logger | None = None,
) -> list[tuple[str, str]]:
    """Pod に残っている会話スレッドを (ID, 見出し) の新しい順で返す。"""
    log = logger or logging.getLogger(__name__)
    normalized, _ = split_runpod_target(base_url)
    timeout = max(1.0, float(timeout_seconds))
    token = _runpod_resolve_token(normalized, api_key, timeout, log)
    try:
        data = _runpod_api(normalized, "/api/v1/chats/", token, timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"runpod_openwebui chat list HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "runpod_openwebui chat list failed "
            f"(Pod が停止しているか URL が古い可能性があります): {exc}"
        ) from exc
    chats: list[tuple[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            chat_id = str(item.get("id") or "").strip()
            if not chat_id:
                continue
            title = str(item.get("title") or "").strip() or "(無題)"
            chats.append((chat_id, title))
    return chats


def create_runpod_chat(
    base_url: str,
    api_key: str,
    model: str,
    *,
    title: str = "Human2KKS",
    timeout_seconds: float = 30.0,
    logger: logging.Logger | None = None,
) -> str:
    """新しい会話スレッドを作り、以後の送信先として控える。"""
    log = logger or logging.getLogger(__name__)
    normalized, _ = split_runpod_target(base_url)
    timeout = max(1.0, float(timeout_seconds))
    token = _runpod_resolve_token(normalized, api_key, timeout, log)
    created = _runpod_api(
        normalized,
        "/api/v1/chats/new",
        token,
        timeout,
        payload={
            "chat": {
                "title": title,
                "models": [model] if model else [],
                "messages": [],
                "history": {"messages": {}, "currentId": None},
            }
        },
    )
    chat_id = str(created.get("id") or "") if isinstance(created, dict) else ""
    if not chat_id:
        raise RuntimeError("runpod_openwebui: 新しい会話を作成できませんでした")
    _runpod_save_chat_id(normalized, chat_id)
    log.info("runpod_chat_created id=%s", chat_id)
    return chat_id


def select_runpod_chat(base_url: str, chat_id: str) -> None:
    """以後の送信先とする会話スレッドを控える。"""
    normalized, _ = split_runpod_target(base_url)
    _runpod_save_chat_id(normalized, str(chat_id or "").strip())


def current_runpod_chat_id(base_url: str) -> str:
    """いま送信先になっている会話スレッドのID。"""
    try:
        normalized, embedded = split_runpod_target(base_url)
    except RuntimeError:
        return ""
    return embedded or _runpod_load_saved_chat_id(normalized)


def list_runpod_models(
    base_url: str,
    api_key: str,
    *,
    timeout_seconds: float = 30.0,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Pod に入っているモデル名を取得する。GUIのモデル選択肢の生成に使う。"""
    log = logger or logging.getLogger(__name__)
    normalized = normalize_runpod_base_url(base_url)
    timeout = max(1.0, float(timeout_seconds))
    token = _runpod_resolve_token(normalized, api_key, timeout, log)

    def _fetch(bearer: str) -> list[str]:
        url = f"{_runpod_webui_root(normalized)}/ollama/api/tags"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {bearer}")
        req.add_header("User-Agent", _RUNPOD_USER_AGENT)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        names: list[str] = []
        for item in models:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("model") or "").strip()
                if name and name not in names:
                    names.append(name)
        return sorted(names)

    try:
        return _fetch(token)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403) and split_runpod_credentials(api_key) is not None:
            token = _runpod_resolve_token(
                normalized, api_key, timeout, log, force_refresh=True
            )
            try:
                return _fetch(token)
            except urllib.error.HTTPError as retry_exc:
                raise RuntimeError(
                    f"runpod_openwebui model list HTTP {retry_exc.code}"
                ) from retry_exc
        raise RuntimeError(f"runpod_openwebui model list HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "runpod_openwebui model list failed "
            f"(Pod が停止しているか URL が古い可能性があります): {exc}"
        ) from exc


# RunPod のプロキシは Cloudflare 配下で、1接続が約100秒を超えると 524 で切られる。
# 17GB級のモデルは初回読み込みだけでこれを超えるため、
# 「読み込みを始めさせる」「載るまで待つ」「本番を送る」を別々の接続に分ける。
_RUNPOD_EDGE_LIMIT_SECONDS = 90.0


def _runpod_loaded_models(base_url: str, token: str, timeout: float) -> set[str]:
    try:
        data = _runpod_api(base_url, "/ollama/api/ps", token, timeout)
    except Exception:
        return set()
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return set()
    return {
        str(item.get("name") or "")
        for item in models
        if isinstance(item, dict) and item.get("name")
    }


def _runpod_ensure_model_loaded(
    base_url: str,
    token: str,
    model: str,
    budget_seconds: float,
    logger: logging.Logger,
) -> bool:
    """モデルがVRAMに載るまで待つ。載っていれば即座に True。

    読み込み要求は Cloudflare に切られる前提で投げ捨て、実際の完了は
    軽い問い合わせで確認する。切断後もサーバー側の読み込みは続くため成立する。
    """
    if not model:
        return False
    if model in _runpod_loaded_models(base_url, token, 20.0):
        return True

    logger.info("runpod_model_warmup_start model=%s", model)
    # 読み込みの引き金。応答は待たない（どうせ切られる）。
    try:
        payload = {"model": model, "prompt": "", "stream": False, "keep_alive": -1}
        _runpod_api(base_url, "/ollama/api/generate", token, 20.0, payload=payload)
    except Exception:
        pass  # 切断・タイムアウトは想定内。読み込み自体は進む。

    deadline = time.monotonic() + max(30.0, float(budget_seconds))
    while time.monotonic() < deadline:
        time.sleep(5.0)
        if model in _runpod_loaded_models(base_url, token, 20.0):
            waited = int(budget_seconds - (deadline - time.monotonic()))
            logger.info("runpod_model_warmup_done model=%s waited=%ds", model, waited)
            return True
    logger.warning("runpod_model_warmup_timeout model=%s", model)
    return False


def _generate_runpod_response(
    text: str,
    *,
    llm_config: LlmRequestConfig,
    logger: logging.Logger,
) -> str:
    base_url, chat_id = split_runpod_target(llm_config.base_url)
    timeout = max(1.0, float(llm_config.timeout_seconds))
    uses_credentials = split_runpod_credentials(llm_config.api_key) is not None

    token = _runpod_resolve_token(base_url, llm_config.api_key, timeout, logger)

    # 会話IDが指定されていなければ、前回控えたものを使う。それも無ければ新規作成。
    # これで発話ごとに別プロセスで起動されても、同じスレッドの続きになる。
    if not chat_id:
        chat_id = _runpod_load_saved_chat_id(base_url)
    api_timeout = min(timeout, 30.0)
    try:
        chat_id, past = _runpod_ensure_chat(
            base_url, chat_id, token, api_timeout, llm_config.model, logger
        )
    except Exception as exc:
        logger.warning(
            "runpod_chat_unavailable reason=%s", sanitize_log_text(exc, max_chars=150)
        )
        chat_id, past = "", []
    if chat_id:
        _runpod_save_chat_id(base_url, chat_id)

    # これまでのやり取り + 今回の発話。system は毎回先頭に置く。
    history: list[dict[str, Any]] = []
    for message in past:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})

    logger.info(
        "llm_runpod_request base_url=%s model=%s auth=%s chat=%s past_turns=%d",
        base_url,
        llm_config.model,
        "credentials" if uses_credentials else "token",
        chat_id or "(なし)",
        len(history),
    )

    # Cloudflare の 100秒制限に本番の要求がぶつからないよう、先に載せておく。
    _runpod_ensure_model_loaded(
        base_url, token, str(llm_config.model or ""), timeout, logger
    )

    def _call(bearer: str) -> str:
        # 生成そのものは速いので、接続ごとの上限を超えないところで打ち切る。
        cfg = replace(
            llm_config,
            base_url=base_url,
            api_key=bearer,
            timeout_seconds=min(float(llm_config.timeout_seconds), _RUNPOD_EDGE_LIMIT_SECONDS),
        )
        return _generate_openai_compatible_response(
            text,
            llm_config=cfg,
            logger=logger,
            label="runpod_openwebui",
            user_agent=_RUNPOD_USER_AGENT,
            history=history,
            suppress_thinking=True,
        )

    answer = ""
    last_error: RuntimeError | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            answer = _call(token)
            last_error = None
            break
        except RuntimeError as exc:
            # 期限切れ(28日)や Pod 作り直しでトークンが無効になった場合は取り直して即再送。
            if uses_credentials and _is_auth_error(exc):
                logger.warning(
                    "runpod_token_refresh reason=%s", sanitize_log_text(exc, max_chars=200)
                )
                token = _runpod_resolve_token(
                    base_url, llm_config.api_key, timeout, logger, force_refresh=True
                )
                last_error = exc
                continue
            # 偶発的な生成失敗は待って送り直す。それ以外は即座に投げる。
            if not _is_retryable_generation_error(exc) or attempt >= _RETRY_ATTEMPTS:
                raise
            logger.warning(
                "runpod_generation_retry attempt=%d/%d reason=%s",
                attempt,
                _RETRY_ATTEMPTS,
                sanitize_log_text(exc, max_chars=200),
            )
            last_error = exc
            time.sleep(_RETRY_WAIT_SECONDS)
    if last_error is not None:
        raise last_error

    _runpod_append_chat(
        base_url,
        chat_id,
        token,
        api_timeout,
        llm_config.model,
        history
        + [{"role": "user", "content": str(text or "")}, {"role": "assistant", "content": answer}],
        logger,
    )
    return answer


# 思考型モデルは答える前に長い独り言を書く。そのまま読み上げると別人の内心が
# 台詞として流れるので、答えだけを取り出す。閉じ記号しか出さない個体もあるため、
# 「開始タグが無くても閉じタグ以前は全部捨てる」方針にしている。
_THINK_BLOCK_PATTERN = re.compile(
    r"<(think|thinking|reasoning)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_THINK_DETAILS_PATTERN = re.compile(
    r"<details\b[^>]*type=[\"']?reasoning[\"']?[^>]*>.*?</details>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_CLOSER_PATTERN = re.compile(r"</(think|thinking|reasoning)\s*>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """モデルの思考部分を落として、実際の返答だけを返す。"""
    body = str(text or "")
    body = _THINK_DETAILS_PATTERN.sub("", body)
    body = _THINK_BLOCK_PATTERN.sub("", body)
    # 開始タグが無いまま閉じタグだけ出る形。最後の閉じタグ以降が答え。
    closers = list(_THINK_CLOSER_PATTERN.finditer(body))
    if closers:
        body = body[closers[-1].end():]
    # 開始タグだけ出て閉じない場合は、答えが無いので元を残す（消し過ぎ防止）。
    stripped = body.strip()
    return stripped if stripped else str(text or "").strip()


# ト書き（丸括弧の中の動作説明）。台詞ではないので読み上げから外す。
# 入れ子にも対応するため正規表現ではなく走査で消す。
_STAGE_DIRECTION_PAIRS = (("(", ")"), ("（", "）"), ("*", "*"))


def _strip_paren_pairs(text: str, opener: str, closer: str) -> str:
    """opener/closer で囲まれた範囲を削る。閉じていない場合は消さない。"""
    if opener not in text:
        return text
    out: list[str] = []
    depth = 0
    index = 0
    same = opener == closer
    while index < len(text):
        char = text[index]
        if same and char == opener:
            depth = 0 if depth else 1
            index += 1
            continue
        if char == opener:
            depth += 1
        elif char == closer and depth:
            depth -= 1
        elif depth == 0:
            out.append(char)
        index += 1
    result = "".join(out)
    # 閉じ忘れで本文まで消えた場合は元を返す（消し過ぎ防止）。
    return result if depth == 0 and result.strip() else text


def strip_stage_directions(text: str) -> str:
    """丸括弧内のト書きを落とし、空行の連続を整える。"""
    body = str(text or "")
    if not body:
        return body
    for opener, closer in _STAGE_DIRECTION_PAIRS:
        body = _strip_paren_pairs(body, opener, closer)
    lines = [line.rstrip() for line in body.splitlines()]
    kept = [line for line in lines if line.strip()]
    return "\n".join(kept).strip()


def _is_auth_error(exc: Exception) -> bool:
    message = str(exc)
    return "HTTP 401" in message or "HTTP 403" in message


# 生成が期待の形式にならず 500 が返ることがある（実測: peg-gemma4 format エラー）。
# 設定不備ではなく偶発なので、黙って諦めず送り直す。
_RETRYABLE_MARKERS = ("HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "does not match the expected")
_RETRY_ATTEMPTS = 3
_RETRY_WAIT_SECONDS = 2.0


def _is_retryable_generation_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def generate_llm_response(
    text: str,
    *,
    llm_config: LlmRequestConfig,
    bridge_config: BridgeConfig,
    logger: logging.Logger,
) -> tuple[str, str]:
    backend = normalize_backend(llm_config.backend)
    if backend == "grok_browser":
        if llm_config.grok_history_enabled:
            try:
                return backend, _generate_grok_history_response(
                    text,
                    llm_config=llm_config,
                    bridge_config=bridge_config,
                    logger=logger,
                )
            except RuntimeError as exc:
                if not llm_config.grok_history_fallback_live:
                    raise
                logger.warning(
                    "grok_history_fallback_live reason=%s",
                    sanitize_log_text(exc, max_chars=500),
                )
                return backend, _generate_grok_live_response(
                    text,
                    bridge_config=bridge_config,
                    logger=logger,
                )
        return backend, _generate_grok_live_response(
            text,
            bridge_config=bridge_config,
            logger=logger,
        )
    if backend == "local_openai":
        return backend, _generate_openai_compatible_response(text, llm_config=llm_config, logger=logger)
    if backend == "runpod_openwebui":
        return backend, _generate_runpod_response(text, llm_config=llm_config, logger=logger)
    raise ValueError(f"unsupported llm backend: {backend}")


def _generate_grok_live_response(
    text: str,
    *,
    bridge_config: BridgeConfig,
    logger: logging.Logger,
) -> str:
    from .browser import connect_existing_debug_chrome
    from .grok_client import send_text, wait_for_response

    driver = connect_existing_debug_chrome(bridge_config.debug_port)
    baseline, stop_before = send_text(driver, bridge_config, text, logger)
    return wait_for_response(driver, bridge_config, logger, baseline, stop_before)


def _is_loopback_http_url(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        return False
    if parsed.hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def _generate_grok_history_response(
    text: str,
    *,
    llm_config: LlmRequestConfig,
    bridge_config: BridgeConfig,
    logger: logging.Logger,
) -> str:
    query = str(text or "").strip()
    if not query:
        raise RuntimeError("grok history search requires non-empty text")
    url = str(
        llm_config.grok_history_search_url
        or bridge_config.history_search_url
        or ""
    ).strip()
    if not _is_loopback_http_url(url):
        raise RuntimeError(f"grok history search URL must be loopback HTTP: {url}")
    configured_timeout = float(llm_config.grok_history_timeout_seconds)
    timeout = max(
        1.0,
        configured_timeout
        if configured_timeout > 0
        else float(bridge_config.history_search_timeout_seconds),
    )
    top_k = max(1, min(100, int(llm_config.grok_history_top_k)))
    payload = {"query": query, "top_k": top_k}
    # 期間指定はサーバ側で絞る。ここで受け取ってから絞ると top_k 件から
    # さらに減るだけで候補が枯渇するため、必ずリクエストに載せる。
    date_from = str(llm_config.grok_history_date_from or "").strip()
    date_to = str(llm_config.grok_history_date_to or "").strip()
    if date_from:
        payload["date_from"] = date_from
    if date_to:
        payload["date_to"] = date_to
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    logger.info(
        "grok_history_request url=%s query_len=%d top_k=%d timeout=%.1f required_terms=%d preferred_terms=%d",
        url,
        len(query),
        top_k,
        timeout,
        len(_split_filter_terms(llm_config.grok_history_response_required_terms)),
        len(_split_filter_terms(llm_config.grok_history_response_preferred_terms)),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = sanitize_log_text(
            exc.read().decode("utf-8", errors="replace") if exc.fp else "",
            max_chars=500,
        )
        raise RuntimeError(f"grok history API HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"grok history API connection failed: {exc}") from exc

    try:
        data: Any = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("grok history API returned invalid JSON") from exc
    if not isinstance(data, dict) or not bool(data.get("ok")):
        detail = data.get("error", "unknown error") if isinstance(data, dict) else "invalid response"
        raise RuntimeError(
            "grok history API search failed: "
            + sanitize_log_text(detail, max_chars=300)
        )

    results = data.get("results")
    candidates = (
        [item for item in results if isinstance(item, dict)][:top_k]
        if isinstance(results, list)
        else []
    )
    if not candidates and isinstance(data.get("selected"), dict):
        candidates = [data["selected"]]
    selected, selection_info = _select_history_candidate(candidates, llm_config)
    assistant_text = str(selected.get("assistant_text") or "")

    trace = {
        "conversation_id": str(selected.get("conversation_id") or ""),
        "user_message_id": int(selected.get("message_id") or 0),
        "assistant_message_ids": [
            int(reply.get("message_id") or 0)
            for reply in selected.get("assistant_replies", [])
            if isinstance(reply, dict)
        ],
        "rank": _candidate_rank(selected),
        "score": _candidate_score(selected),
        "user_text_chars": len(str(selected.get("user_text") or "")),
        "assistant_text_chars": len(assistant_text),
        **selection_info,
    }
    logger.info("grok_history_match=%s", json.dumps(trace, ensure_ascii=False))
    return assistant_text


def _split_filter_terms(value: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_line in str(value or "").splitlines():
        term = raw_line.strip()
        folded = term.casefold()
        if term and folded not in seen:
            seen.add(folded)
            terms.append(term)
    return terms


def _candidate_score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_rank(candidate: dict[str, Any]) -> int:
    try:
        return max(0, int(candidate.get("rank") or 0))
    except (TypeError, ValueError):
        return 0


def _select_history_candidate(
    candidates: list[dict[str, Any]],
    llm_config: LlmRequestConfig,
) -> tuple[dict[str, Any], dict[str, int | str]]:
    required_terms = _split_filter_terms(
        llm_config.grok_history_response_required_terms
    )
    preferred_terms = _split_filter_terms(
        llm_config.grok_history_response_preferred_terms
    )
    required_mode = str(llm_config.grok_history_required_match_mode or "any").strip().lower()
    if required_mode not in ("any", "all"):
        required_mode = "any"

    min_score = float(llm_config.grok_history_min_score)
    eligible: list[tuple[dict[str, Any], int, float, int]] = []
    for candidate in candidates:
        assistant_text = str(candidate.get("assistant_text") or "")
        if not assistant_text.strip():
            continue
        score = _candidate_score(candidate)
        if min_score > -1.0 and score < min_score:
            continue

        folded_text = assistant_text.casefold()
        required_matches = [term.casefold() in folded_text for term in required_terms]
        if required_matches:
            required_ok = all(required_matches) if required_mode == "all" else any(required_matches)
            if not required_ok:
                continue

        preferred_count = sum(
            1 for term in preferred_terms if term.casefold() in folded_text
        )
        eligible.append(
            (candidate, preferred_count, score, _candidate_rank(candidate))
        )

    if not eligible:
        raise RuntimeError(
            "grok history search returned no candidate after response filters "
            f"(received={len(candidates)} required_terms={len(required_terms)} "
            f"min_score={min_score:.2f})"
        )

    eligible.sort(key=lambda item: (-item[1], -item[2], item[3]))
    selection_mode = str(llm_config.grok_history_selection_mode or "best").strip().lower()
    if selection_mode not in ("best", "random"):
        selection_mode = "best"

    selection_pool = eligible
    highest_preferred_count = eligible[0][1]
    if highest_preferred_count > 0:
        selection_pool = [
            item for item in eligible if item[1] == highest_preferred_count
        ]
    selected_item = (
        random.choice(selection_pool) if selection_mode == "random" else eligible[0]
    )
    return selected_item[0], {
        "received_candidates": len(candidates),
        "eligible_candidates": len(eligible),
        "selection_pool_candidates": len(selection_pool),
        "preferred_term_matches": selected_item[1],
        "selection_mode": selection_mode,
    }


def _generate_openai_compatible_response(
    text: str,
    *,
    llm_config: LlmRequestConfig,
    logger: logging.Logger,
    label: str = "local_openai",
    user_agent: str = "",
    history: list[dict[str, str]] | None = None,
    suppress_thinking: bool = False,
) -> str:
    base_url = str(llm_config.base_url or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError(f"{label} requires llm_base_url")
    model = str(llm_config.model or "").strip()
    if not model:
        raise RuntimeError(f"{label} requires llm_model")

    messages: list[dict[str, str]] = []
    system_prompt = str(llm_config.system_prompt or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for past in history or []:  # これまでのやり取り（会話スレッド利用時のみ入る）
        messages.append({"role": str(past.get("role")), "content": str(past.get("content"))})
    messages.append({"role": "user", "content": str(text or "")})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(llm_config.temperature),
        "max_tokens": max(1, int(llm_config.max_tokens)),
        "stream": False,
    }
    if suppress_thinking:
        # 思考にもtokenを使うため、抑えられる分は本文に回す。
        # 対応していないサーバーは無視するだけなので付けても害はない。
        payload["think"] = False
        payload["chat_template_kwargs"] = {"thinking": False}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{base_url}/chat/completions"
    req = urllib.request.Request(url, data=raw, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    if user_agent:
        req.add_header("User-Agent", user_agent)
    api_key = str(llm_config.api_key or "").strip()
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    timeout = max(1.0, float(llm_config.timeout_seconds))
    logger.info(
        "llm_%s_request base_url=%s model=%s text_len=%d timeout=%.1f",
        label,
        base_url,
        model,
        len(text or ""),
        timeout,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"{label} HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        hint = (
            " (Pod が停止しているか URL が古い可能性があります)"
            if label == "runpod_openwebui"
            else ""
        )
        raise RuntimeError(f"{label} connection failed{hint}: {exc}") from exc

    try:
        data: Any = json.loads(body)
    except Exception as exc:
        raise RuntimeError(f"{label} returned invalid JSON: {body[:500]}") from exc

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{label} response has no choices: {body[:500]}")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"{label} invalid first choice: {body[:500]}")
    message = first.get("message")
    content = ""
    if isinstance(message, dict):
        content = str(message.get("content") or "").strip()
    if not content:
        content = str(first.get("text") or "").strip()
    if not content:
        # 思考型モデルは reasoning を別欄に返す。上限に達すると思考だけで
        # 終わり本文が空になるため、原因が分かる文言にする。
        reasoning_len = 0
        if isinstance(message, dict):
            reasoning_len = len(str(message.get("reasoning") or ""))
        if reasoning_len and str(first.get("finish_reason") or "") == "length":
            raise RuntimeError(
                f"{label}: 思考だけで max_tokens を使い切りました"
                f"(思考{reasoning_len}文字)。max_tokens を増やしてください"
            )
        raise RuntimeError(f"{label} response content is empty: {body[:500]}")

    answer = strip_thinking(content)
    if len(answer) != len(content):
        logger.info(
            "llm_%s_thinking_stripped raw=%d kept=%d", label, len(content), len(answer)
        )
    logger.info("llm_%s_response len=%d", label, len(answer))
    return answer
