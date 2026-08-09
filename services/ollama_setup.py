"""Ollama本体と埋め込みモデルの有無を調べ、必要なら取得する。

取り込み(埋め込み)は Ollama とモデルが両方そろっていないと必ず失敗する。
今までは失敗して初めて分かる作りで、しかもエラーが
"embed failed (exit N). Ollama is running?" としか出ないため、
モデルが無いだけなのに起動を疑ってしまう。ここで先に切り分ける。

本体のインストールは自動でやらない。別ソフトを黙って入れるのは避ける。
モデルの取得は数GBあるので、呼び出し側で確認を取ってから実行する。
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from typing import Callable, Optional

# 埋め込みに使うモデル。ollama_vectors.py の OLLAMA_MODEL と揃える。
REQUIRED_MODEL = "bge-m3"
REQUIRED_MODEL_TAG = "bge-m3:latest"

DOWNLOAD_URL = "https://ollama.com"


def _endpoint(settings: Optional[dict]) -> str:
    if settings:
        value = str(settings.get("grok_history_ollama_endpoint", "") or "").strip()
        if value:
            return value.rstrip("/")
    return "http://127.0.0.1:11434"


def resolve_exe(settings: Optional[dict] = None) -> Optional[str]:
    """ollama.exe の場所。設定优先、無ければ標準の場所を探す。"""
    from services import grok_history_server

    configured = ""
    if settings:
        configured = str(settings.get("grok_history_ollama_exe", "") or "")

    path = grok_history_server.resolve_ollama(configured)
    return str(path) if path is not None else None


def is_installed(settings: Optional[dict] = None) -> bool:
    """Ollama本体があるか。実行ファイルが見つかるか、待ち受けていればOK。"""
    if resolve_exe(settings) is not None:
        return True

    # 別マシンのOllamaを指している場合、実行ファイルは無くても使える。
    return is_running(settings)


def is_running(settings: Optional[dict] = None) -> bool:
    try:
        with urllib.request.urlopen(_endpoint(settings) + "/api/tags", timeout=2.0):
            return True
    except Exception:
        return False


def has_model(settings: Optional[dict] = None) -> Optional[bool]:
    """モデルが入っているか。Ollamaに繋がらない場合は None（判定不能）。"""
    try:
        with urllib.request.urlopen(_endpoint(settings) + "/api/tags", timeout=5.0) as res:
            data = json.loads(res.read().decode("utf-8", "replace"))
    except Exception:
        return None

    for entry in data.get("models", []) or []:
        name = str(entry.get("name", "") or "")
        if name == REQUIRED_MODEL_TAG or name.split(":")[0] == REQUIRED_MODEL:
            return True
    return False


def pull_model(
    settings: Optional[dict] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """モデルを取得する。戻り値は失敗理由（成功時 None）。数GBあるので時間が掛かる。"""

    def log(message: str) -> None:
        if emit is not None:
            emit(message)

    exe = resolve_exe(settings)
    if exe is None:
        return "ollama.exe が見つかりません"

    log(f"[ollama] {REQUIRED_MODEL} を取得します（数GBあります）")
    try:
        proc = subprocess.Popen(
            [exe, "pull", REQUIRED_MODEL],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as ex:
        return f"取得を開始できません: {ex}"

    last = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        # 進捗行は毎行出すと溢れるので、内容が変わった時だけ出す。
        if line and line != last:
            last = line
            log(f"[ollama] {line}")

    code = proc.wait()
    if code != 0:
        return f"取得に失敗しました (exit {code})"

    log(f"[ollama] {REQUIRED_MODEL} の取得が終わりました")
    return None


def check(settings: Optional[dict] = None) -> tuple[str, str]:
    """取り込み前の判定。(状態, 表示用メッセージ) を返す。

    状態は "ok" / "no_ollama" / "no_model" / "unknown" のいずれか。
    """
    if not is_installed(settings):
        return (
            "no_ollama",
            "【この機能について】\n"
            "Grokとの会話履歴が溜まっていれば、それを丸ごと取り込んで、\n"
            "過去に自分が交わした会話を再利用した疑似会話ができます。\n"
            "履歴が膨大なほど、それらしい返事が返ります。\n"
            "毎回AIに考えさせるより速く、口調も過去の自分たちのままになります。\n"
            "\n"
            "【なぜ Ollama が要るのか】\n"
            "「昨日なに食べた？」と「昨日の晩ごはんは？」が似ている、と機械に\n"
            "判断させるには、文章を数字の並びに変換する必要があります。\n"
            "その変換を Ollama というソフトに任せています。\n"
            "\n"
            "【今の状態】\n"
            "Ollama が見つかりません。\n"
            "\n"
            "※ この機能を使わないなら Ollama は不要です。\n"
            "　 会話・読み上げ・字幕などは Ollama 無しで動きます。\n"
            "\n"
            "ダウンロードページ (" + DOWNLOAD_URL + ") を開きますか？",
        )

    model = has_model(settings)
    if model is True:
        return ("ok", "")
    if model is False:
        return (
            "no_model",
            "【この機能について】\n"
            "Grokとの会話履歴を取り込んで、過去の会話を再利用した疑似会話をします。\n"
            "履歴が膨大なほど、それらしい返事が返ります。\n"
            "\n"
            "【あと1つ足りません】\n"
            "文章を数字に変換するモデル「" + REQUIRED_MODEL + "」が入っていません。\n"
            "これが無いと、会話が似ているかどうかを判定できません。\n"
            "\n"
            "・ダウンロードは数GBあります（回線によっては数分〜十数分）\n"
            "・1回入れれば、次回からは聞かれません\n"
            "・取得中はこの画面が固まります。進み具合はログに出ます\n"
            "\n"
            "今すぐインストールしますか？",
        )

    # Ollama自体は入っているが待ち受けていない。起動処理が別途走るので続行させる。
    return ("unknown", "")
