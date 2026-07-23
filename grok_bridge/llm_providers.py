from __future__ import annotations

import ipaddress
import json
import logging
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = ""
    api_key: str = "lm-studio"
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_seconds: float = 120.0


def compose_llm_input(text: str, always_append_text: str) -> str:
    """Append the configured fixed phrase only to the text sent to the LLM/search."""
    base = str(text or "").rstrip()
    suffix = str(always_append_text or "").strip()
    if not suffix:
        return str(text or "")
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
    }
    if token not in aliases:
        raise ValueError(f"unsupported llm backend: {value}")
    return aliases[token]


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
) -> str:
    base_url = str(llm_config.base_url or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("local_openai requires llm_base_url")
    model = str(llm_config.model or "").strip()
    if not model:
        raise RuntimeError("local_openai requires llm_model")

    messages: list[dict[str, str]] = []
    system_prompt = str(llm_config.system_prompt or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": str(text or "")})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(llm_config.temperature),
        "max_tokens": max(1, int(llm_config.max_tokens)),
        "stream": False,
    }
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{base_url}/chat/completions"
    req = urllib.request.Request(url, data=raw, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    api_key = str(llm_config.api_key or "").strip()
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    timeout = max(1.0, float(llm_config.timeout_seconds))
    logger.info(
        "llm_local_openai_request base_url=%s model=%s text_len=%d timeout=%.1f",
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
        raise RuntimeError(f"local_openai HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"local_openai connection failed: {exc}") from exc

    try:
        data: Any = json.loads(body)
    except Exception as exc:
        raise RuntimeError(f"local_openai returned invalid JSON: {body[:500]}") from exc

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"local_openai response has no choices: {body[:500]}")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"local_openai invalid first choice: {body[:500]}")
    message = first.get("message")
    content = ""
    if isinstance(message, dict):
        content = str(message.get("content") or "").strip()
    if not content:
        content = str(first.get("text") or "").strip()
    if not content:
        raise RuntimeError(f"local_openai response content is empty: {body[:500]}")

    logger.info("llm_local_openai_response len=%d", len(content))
    return content
