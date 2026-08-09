from __future__ import annotations

import sys
from typing import Callable, Optional

from PyQt6.QtWidgets import QApplication, QMainWindow

from config.constants import MUTEX_NAME
from core.single_instance import acquire_single_instance


def run_gui(
    window_factory: Callable[[], QMainWindow],
    *,
    argv: Optional[list[str]] = None,
    mutex_name: str = MUTEX_NAME,
) -> int:
    if not acquire_single_instance(mutex_name):
        print("[info] Already running.")
        return 0

    # Grok履歴ベクター検索API(8877)をGUIと同じ寿命で動かす。
    # 起動できなくてもGUI自体は動かす(ベクター返答が使えないだけ)。
    from services import grok_history_server

    reason = grok_history_server.start()
    if reason:
        print(f"[warn] grok history server not started: {reason}")
    else:
        print("[info] grok history server ready (127.0.0.1:8877)")

    try:
        app = QApplication(sys.argv if argv is None else argv)
        window = window_factory()
        window.show()
        return app.exec()
    finally:
        grok_history_server.stop()


def run_gui_default(argv: Optional[list[str]] = None) -> int:
    from gui.main_window import MainWindow

    return run_gui(MainWindow, argv=argv, mutex_name=MUTEX_NAME)
