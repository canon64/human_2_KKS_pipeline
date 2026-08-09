from __future__ import annotations

from pathlib import Path
import argparse
import sys

from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from scripts.grok_export_browser.ollama_vectors import (
    DEFAULT_INDEX_PATH,
    DEFAULT_RAW_DB,
    DEFAULT_VECTOR_DB,
)
from scripts.grok_export_browser.user_turn_search import UserTurnHit, UserTurnSearchService


class SearchWorker(QThread):
    results_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        service: UserTurnSearchService,
        query: str,
        top_k: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.query = query
        self.top_k = top_k

    def run(self) -> None:
        try:
            self.results_ready.emit(self.service.search(self.query, top_k=self.top_k))
        except Exception as exc:
            self.error_occurred.emit(str(exc))


def _snippet(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


class GrokVectorSearchWindow(QMainWindow):
    def __init__(self, service: UserTurnSearchService) -> None:
        super().__init__()
        self.service = service
        self.settings = QSettings("APIScripts", "GrokUserVectorSearch")
        self._worker: SearchWorker | None = None
        self._results: list[UserTurnHit] = []
        self.setWindowTitle("Grok ユーザー発言ベクター検索")
        self.setMinimumSize(980, 680)
        self.resize(1420, 880)
        self._build_ui()
        self._restore_settings()
        stats = self.service.stats()
        self.status.setText(
            f"準備完了 — ユーザー発言 {stats['unique_user_messages']:,}件 / "
            f"検索チャンク {stats['user_chunks']:,}件 / "
            f"直接応答あり {stats['user_messages_with_direct_assistant']:,}件"
        )
        self.query.setFocus()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("Grok ユーザー発言ベクター検索")
        title_font = QFont(self.font())
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        description = QLabel(
            "入力文に意味が近いユーザー発言だけを検索し、その発言へのGrokの直接応答を対で表示します。"
        )
        description.setWordWrap(True)
        root.addWidget(description)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("検索文"))
        self.query = QLineEdit()
        self.query.setPlaceholderText("例: ローカルLLMで検索システムを作る方法")
        self.query.setClearButtonEnabled(True)
        self.query.returnPressed.connect(self.run_search)
        search_row.addWidget(self.query, 1)
        search_row.addWidget(QLabel("上位"))
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 200)
        self.top_k.setValue(20)
        self.top_k.setSuffix(" 件")
        search_row.addWidget(self.top_k)
        self.search_button = QPushButton("ベクター検索")
        self.search_button.setDefault(True)
        self.search_button.clicked.connect(self.run_search)
        search_row.addWidget(self.search_button)
        root.addLayout(search_row)

        self.status = QLabel("索引を読み込んでいます…")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(self.main_splitter, 1)

        result_group = QGroupBox("検索結果 — クリックすると下に全文を表示")
        result_layout = QVBoxLayout(result_group)
        self.result_table = QTableWidget(0, 6)
        self.result_table.setHorizontalHeaderLabels(
            ["順位", "類似度", "日時", "会話タイトル", "該当ユーザー発言", "Grokの応答"]
        )
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setWordWrap(False)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.result_table.itemSelectionChanged.connect(self.show_selected_result)
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        result_layout.addWidget(self.result_table)
        self.main_splitter.addWidget(result_group)

        self.detail_splitter = QSplitter(Qt.Orientation.Horizontal)
        user_group = QGroupBox("該当したユーザー発言（全文）")
        user_layout = QVBoxLayout(user_group)
        self.match_info = QLabel("検索結果を選択してください。")
        self.match_info.setWordWrap(True)
        self.match_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        user_layout.addWidget(self.match_info)
        self.user_text = QTextEdit()
        self.user_text.setReadOnly(True)
        self.user_text.setPlaceholderText("該当したユーザー発言がここに表示されます。")
        user_layout.addWidget(self.user_text, 1)
        self.detail_splitter.addWidget(user_group)

        assistant_group = QGroupBox("その発言に反応したアシスタント本文（全文）")
        assistant_layout = QVBoxLayout(assistant_group)
        self.assistant_info = QLabel("直接応答を表示します。")
        self.assistant_info.setWordWrap(True)
        self.assistant_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        assistant_layout.addWidget(self.assistant_info)
        self.assistant_text = QTextEdit()
        self.assistant_text.setReadOnly(True)
        self.assistant_text.setPlaceholderText("Grokの応答本文がここに表示されます。")
        assistant_layout.addWidget(self.assistant_text, 1)
        self.detail_splitter.addWidget(assistant_group)
        self.main_splitter.addWidget(self.detail_splitter)

        self.main_splitter.setSizes([410, 390])
        self.detail_splitter.setSizes([700, 700])

    def _restore_settings(self) -> None:
        self.top_k.setValue(int(self.settings.value("top_k", 20)))
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        main_state = self.settings.value("main_splitter")
        if main_state is not None:
            self.main_splitter.restoreState(main_state)
        detail_state = self.settings.value("detail_splitter")
        if detail_state is not None:
            self.detail_splitter.restoreState(detail_state)

    def run_search(self) -> None:
        query = self.query.text().strip()
        if not query:
            self.status.setText("検索文を入力してください。")
            self.query.setFocus()
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self.settings.setValue("top_k", self.top_k.value())
        self.search_button.setEnabled(False)
        self.top_k.setEnabled(False)
        self.status.setText("Ollamaで検索文をベクター化して検索中…")
        self._worker = SearchWorker(self.service, query, self.top_k.value(), self)
        self._worker.results_ready.connect(self.apply_results)
        self._worker.error_occurred.connect(self.show_error)
        self._worker.finished.connect(self.search_finished)
        self._worker.start()

    def search_finished(self) -> None:
        worker = self._worker
        self._worker = None
        self.search_button.setEnabled(True)
        self.top_k.setEnabled(True)
        if worker is not None:
            worker.deleteLater()

    def show_error(self, message: str) -> None:
        self.status.setText(f"検索エラー: {message}")

    def apply_results(self, rows: object) -> None:
        self._results = list(rows) if isinstance(rows, (list, tuple)) else []
        self.result_table.setRowCount(len(self._results))
        for row_index, hit in enumerate(self._results):
            reply_preview = hit.assistant_text or "（直接応答なし）"
            values = [
                str(hit.rank),
                f"{hit.score:.4f}",
                hit.create_iso,
                hit.conversation_title,
                _snippet(hit.user_text),
                _snippet(reply_preview),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (0, 1):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip(value)
                self.result_table.setItem(row_index, column, item)
        if self._results:
            with_reply = sum(bool(hit.assistant_replies) for hit in self._results)
            self.status.setText(
                f"検索結果 {len(self._results)}件 — 直接応答あり {with_reply}件。"
                "行をクリックすると両方の全文を表示します。"
            )
            self.result_table.selectRow(0)
        else:
            self.status.setText("検索結果はありません。")
            self.clear_details()

    def show_selected_result(self) -> None:
        row = self.result_table.currentRow()
        if row < 0 or row >= len(self._results):
            return
        hit = self._results[row]
        self.match_info.setText(
            f"順位 {hit.rank} / 類似度 {hit.score:.4f} / {hit.create_iso or '日時不明'}\n"
            f"会話: {hit.conversation_title or '（タイトルなし）'}\n"
            f"一致チャンク: #{hit.matched_chunk_id} — {_snippet(hit.matched_chunk_text, 260)}"
        )
        self.user_text.setPlainText(hit.user_text)
        if hit.assistant_replies:
            sections = []
            for index, reply in enumerate(hit.assistant_replies, start=1):
                if len(hit.assistant_replies) > 1:
                    sections.append(f"--- 応答 {index} / {reply.create_iso or '日時不明'} ---\n{reply.text}")
                else:
                    sections.append(reply.text)
            self.assistant_text.setPlainText("\n\n".join(sections))
            self.assistant_info.setText(
                f"このユーザー発言を parent とするassistant応答: {len(hit.assistant_replies)}件"
            )
        else:
            self.assistant_text.setPlainText("（この発言への直接応答本文は保存されていません）")
            self.assistant_info.setText("直接応答: 0件")

    def clear_details(self) -> None:
        self.match_info.setText("検索結果を選択してください。")
        self.user_text.clear()
        self.assistant_info.setText("直接応答を表示します。")
        self.assistant_text.clear()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.status.setText("検索処理が完了してから閉じてください。")
            event.ignore()
            return
        self.settings.setValue("top_k", self.top_k.value())
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("main_splitter", self.main_splitter.saveState())
        self.settings.setValue("detail_splitter", self.detail_splitter.saveState())
        super().closeEvent(event)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grok user-message vector-search GUI")
    parser.add_argument("--vector-db", type=Path, default=DEFAULT_VECTOR_DB)
    parser.add_argument("--raw-db", type=Path, default=DEFAULT_RAW_DB)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--smoke-screenshot", type=Path)
    parser.add_argument("--smoke-query", default="ローカルAIで文章を検索する方法")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("Grok User Vector Search")
    app.setFont(QFont("Yu Gothic UI", 10))
    service = UserTurnSearchService(
        vector_db=args.vector_db,
        raw_db=args.raw_db,
        index_path=args.index,
    )
    window = GrokVectorSearchWindow(service)
    window.show()
    if args.smoke_screenshot is not None:
        window.apply_results(service.search(args.smoke_query, top_k=3))
        app.processEvents()
        args.smoke_screenshot.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(args.smoke_screenshot)):
            raise RuntimeError(f"Could not save GUI screenshot: {args.smoke_screenshot}")
        print(f"smoke_screenshot={args.smoke_screenshot}")
        window.close()
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
