"""Grok履歴ベクター検索API(8877)を、GUIの寿命に紐付けて起動・停止する。

GUIが落ちたらサーバも必ず落とす。通常終了は stop() で止めるが、
タスクマネージャからの強制終了や親プロセスのクラッシュでは atexit も finally も走らない。
そこで Windows の Job Object に子を入れ、ジョブハンドルが閉じた時点で
OS 側が子を kill する形にしている(JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)。
親プロセスが死ねばハンドルは必ず閉じるので、取りこぼしが無い。
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import socket
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8877

# Ollama。8877の検索も履歴更新の埋め込みも、これが無いと動かない。
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
_OLLAMA_CANDIDATES = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
    Path(r"C:\Program Files\Ollama\ollama.exe"),
)

# grok_history/ 以下にサーバ本体とデータを同梱している。
_ROOT = Path(__file__).resolve().parents[1]
GROK_HISTORY_ROOT = _ROOT / "grok_history"

# 同梱 python には faiss が無いため、索引検索できる python を順に探す。
_PYTHON_CANDIDATES = (
    _ROOT / "python" / "python.exe",
    Path(sys.executable),
)

_process: Optional[subprocess.Popen] = None
_job_handle = None
_ollama_process: Optional[subprocess.Popen] = None


def load_settings() -> dict:
    """config.json からこのサービス向けの設定だけ拾う。

    サーバ起動はGUI構築より前に走るので AppConfig は使えない。
    ここでは config.json を直接読み、無い項目は既定値で埋める。
    """
    defaults = {
        "grok_history_autostart": True,
        "grok_history_api_port": DEFAULT_PORT,
        "grok_history_ollama_autostart": True,
        "grok_history_ollama_exe": "",
        "grok_history_ollama_endpoint": f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
        "grok_history_ollama_model": "bge-m3:latest",
    }
    config_file = _ROOT / "config.json"
    if not config_file.is_file():
        return defaults
    try:
        with config_file.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return defaults
    if not isinstance(raw, dict):
        return defaults
    for key in defaults:
        if key in raw and raw[key] not in (None, ""):
            defaults[key] = raw[key]
    return defaults


def _endpoint_host_port(endpoint: str) -> tuple[str, int]:
    """'http://host:port' を (host, port) に分解する。壊れていれば既定へ落とす。"""
    try:
        parsed = urlsplit(str(endpoint or ""))
        host = parsed.hostname or OLLAMA_HOST
        port = int(parsed.port or OLLAMA_PORT)
        return host, port
    except (ValueError, TypeError):
        return OLLAMA_HOST, OLLAMA_PORT


def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _has_faiss(python_exe: Path) -> bool:
    if not python_exe.is_file():
        return False
    try:
        completed = subprocess.run(
            [str(python_exe), "-c", "import faiss"],
            capture_output=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def resolve_python() -> Optional[Path]:
    """faiss が使える python を返す。明示指定があればそれを優先する。"""
    override = os.environ.get("H2K_GROK_HISTORY_PYTHON", "").strip()
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    for candidate in _PYTHON_CANDIDATES:
        if _has_faiss(candidate):
            return candidate
    return None


def resolve_ollama(configured: str = "") -> Optional[Path]:
    """設定 > 環境変数 > 既定の探索場所、の順に ollama.exe を決める。"""
    for override in (configured, os.environ.get("H2K_OLLAMA_EXE", "")):
        token = str(override or "").strip()
        if token:
            candidate = Path(token)
            return candidate if candidate.is_file() else None
    for candidate in _OLLAMA_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def start_ollama(
    timeout_seconds: float = 30.0,
    settings: Optional[dict] = None,
) -> Optional[str]:
    """Ollama を起動する。戻り値は失敗理由(成功時は None)。

    既に待ち受けていれば何もしない。デスクトップ版が常駐している場合や
    ユーザーが手で起動している場合を奪わないため。
    自分で起動した場合だけ Job Object に入れて GUI と道連れにする。
    """
    global _ollama_process

    settings = settings if settings is not None else load_settings()
    host, port = _endpoint_host_port(settings.get("grok_history_ollama_endpoint", ""))

    if _is_port_open(host, port):
        return None

    if not bool(settings.get("grok_history_ollama_autostart", True)):
        return f"ollama not running at {host}:{port} (autostart disabled)"

    # 別マシンのOllamaを指している場合、こちらから起動はできない。
    if host not in ("127.0.0.1", "localhost", "::1"):
        return f"ollama at {host} is remote; start it there"

    exe = resolve_ollama(str(settings.get("grok_history_ollama_exe", "")))
    if exe is None:
        return "ollama.exe not found (set it in the GUI or H2K_OLLAMA_EXE)"

    try:
        _ollama_process = subprocess.Popen(
            [str(exe), "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        _ollama_process = None
        return f"failed to start ollama: {exc}"

    # 起動直後は待ち受けが立ち上がっていないので、開くまで待つ。
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    while time.monotonic() < deadline:
        if _is_port_open(host, port):
            return None
        if _ollama_process.poll() is not None:
            return f"ollama exited immediately (code {_ollama_process.returncode})"
        time.sleep(0.5)
    return f"ollama did not open {port} within {timeout_seconds:.0f}s"


def _create_kill_on_close_job():
    """子を道連れにする Job Object を作る。失敗しても致命ではないので None を返す。"""
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)  # ExtendedLimitInformation
        ):
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


def _assign_to_job(job, pid: int) -> bool:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x001F0FFF, False, pid)  # PROCESS_ALL_ACCESS
        if not handle:
            return False
        try:
            return bool(kernel32.AssignProcessToJobObject(job, handle))
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return False


def start(host: str = DEFAULT_HOST, port: Optional[int] = None) -> Optional[str]:
    """サーバを起動する。戻り値は失敗理由(成功時は None)。"""
    global _process, _job_handle

    settings = load_settings()
    if port is None:
        try:
            port = int(settings.get("grok_history_api_port", DEFAULT_PORT))
        except (TypeError, ValueError):
            port = DEFAULT_PORT

    if os.environ.get("H2K_GROK_HISTORY_AUTOSTART", "").strip() == "0":
        return "autostart disabled by H2K_GROK_HISTORY_AUTOSTART=0"
    if not bool(settings.get("grok_history_autostart", True)):
        return "autostart disabled in settings"

    # 検索も履歴更新も Ollama の埋め込みに依存するので、8877 より先に用意する。
    # ここが失敗しても 8877 は立てる(索引検索以外は動くため)。理由は呼び出し側が拾う。
    ollama_reason = start_ollama(settings=settings)

    if _is_port_open(host, port):
        # 既に誰かが立てている。奪わずにそのまま使う。
        return f"ollama: {ollama_reason}" if ollama_reason else None

    if not GROK_HISTORY_ROOT.is_dir():
        return f"grok_history not found: {GROK_HISTORY_ROOT}"

    python_exe = resolve_python()
    if python_exe is None:
        return "no python with faiss (set H2K_GROK_HISTORY_PYTHON)"

    env = dict(os.environ)
    env["API_SCRIPTS_ROOT"] = str(GROK_HISTORY_ROOT)
    env["PYTHONPATH"] = str(GROK_HISTORY_ROOT)
    env["PYTHONUTF8"] = "1"

    try:
        _process = subprocess.Popen(
            [
                str(python_exe),
                "-m",
                "scripts.grok_export_browser.user_turn_api",
                "--serve",
                "--host",
                str(host),
                "--port",
                str(port),
                "--endpoint",
                str(settings.get("grok_history_ollama_endpoint", "")),
                "--ollama-model",
                str(settings.get("grok_history_ollama_model", "")),
            ],
            cwd=str(GROK_HISTORY_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        _process = None
        return f"failed to start: {exc}"

    _job_handle = _create_kill_on_close_job()
    if _job_handle:
        _assign_to_job(_job_handle, _process.pid)
        # 自分で起動した Ollama も同じジョブへ。既存の Ollama を使った場合は
        # _ollama_process が None なので、他人のプロセスを巻き込むことはない。
        if _ollama_process is not None and _ollama_process.poll() is None:
            _assign_to_job(_job_handle, _ollama_process.pid)

    atexit.register(stop)
    return f"ollama: {ollama_reason}" if ollama_reason else None


def _terminate(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    except OSError:
        pass


def stop() -> None:
    global _process, _job_handle, _ollama_process

    process, _process = _process, None
    _terminate(process)

    # 自分で起動した Ollama だけ止める。既存のものには触らない。
    ollama, _ollama_process = _ollama_process, None
    _terminate(ollama)

    job, _job_handle = _job_handle, None
    if job:
        try:
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)
        except Exception:
            pass


def is_running() -> bool:
    return _process is not None and _process.poll() is None
