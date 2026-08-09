from __future__ import annotations

from PyQt6.QtCore import QDate, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class _NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class _NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


def build_vector_response_tab(window) -> None:
    """Build the Grok history/vector response settings tab on ``window``."""
    inner = QWidget()
    scroll = QScrollArea()
    scroll.setWidget(inner)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    window.tabs.addTab(scroll, "ベクター返答")

    layout = QVBoxLayout(inner)
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    layout.addLayout(form)

    window.grok_history_enabled_chk = QCheckBox("Grok履歴ベクター返答を使う")
    window.grok_history_enabled_chk.setChecked(True)
    window.grok_history_enabled_chk.setToolTip(
        "ONならユーザー発言をベクター検索し、対応するassistant返答を再生します。"
    )
    form.addRow("動作", window.grok_history_enabled_chk)

    window.grok_history_search_url_edit = QLineEdit("http://127.0.0.1:8877/search")
    window.grok_history_search_url_edit.setPlaceholderText("http://127.0.0.1:8877/search")
    window.grok_history_search_url_edit.setToolTip(
        "安全のためlocalhost/127.0.0.1等のループバックHTTPだけを許可します。"
    )
    form.addRow("検索API", window.grok_history_search_url_edit)

    search_row = QHBoxLayout()
    window.grok_history_top_k_spin = _NoWheelSpinBox()
    window.grok_history_top_k_spin.setRange(1, 100)
    window.grok_history_top_k_spin.setValue(10)
    window.grok_history_top_k_spin.setToolTip("ベクター検索から受け取る上位候補数")

    window.grok_history_selection_mode_combo = _NoWheelComboBox()
    window.grok_history_selection_mode_combo.addItem("最上位を選ぶ", "best")
    window.grok_history_selection_mode_combo.addItem("候補からランダム", "random")
    window.grok_history_selection_mode_combo.setToolTip(
        "ランダム時は、絞り込み後の候補から1件を選びます。優先語一致候補があればその候補群を先に使います。"
    )

    window.grok_history_min_score_spin = _NoWheelDoubleSpinBox()
    window.grok_history_min_score_spin.setRange(-1.0, 1.0)
    window.grok_history_min_score_spin.setDecimals(2)
    window.grok_history_min_score_spin.setSingleStep(0.05)
    window.grok_history_min_score_spin.setValue(-1.0)
    window.grok_history_min_score_spin.setSpecialValueText("無効")
    window.grok_history_min_score_spin.setToolTip(
        "この類似度未満を除外します。-1.00（無効）ならスコアでは除外しません。"
    )

    search_row.addWidget(QLabel("候補数"))
    search_row.addWidget(window.grok_history_top_k_spin)
    search_row.addWidget(QLabel("選択"))
    search_row.addWidget(window.grok_history_selection_mode_combo, 1)
    search_row.addWidget(QLabel("最低スコア"))
    search_row.addWidget(window.grok_history_min_score_spin)
    search_widget = QWidget()
    search_widget.setLayout(search_row)
    form.addRow("候補選択", search_widget)

    window.grok_history_timeout_spin = _NoWheelDoubleSpinBox()
    window.grok_history_timeout_spin.setRange(1.0, 120.0)
    window.grok_history_timeout_spin.setDecimals(1)
    window.grok_history_timeout_spin.setValue(30.0)
    window.grok_history_timeout_spin.setSuffix(" 秒")
    form.addRow("検索タイムアウト", window.grok_history_timeout_spin)

    window.grok_history_fallback_live_chk = QCheckBox(
        "該当候補なし／検索失敗時はライブGrokへ送る"
    )
    window.grok_history_fallback_live_chk.setChecked(False)
    form.addRow("フォールバック", window.grok_history_fallback_live_chk)

    window.tts_line_break_target_spin = _NoWheelSpinBox()
    window.tts_line_break_target_spin.setRange(10, 100)
    window.tts_line_break_target_spin.setValue(80)
    window.tts_line_break_target_spin.setSuffix(" 文字")
    window.tts_line_break_target_spin.setToolTip(
        "この文字数までを後ろから探し、最後の「。」「！」「!」または各種ハートで改行します。"
        "なければ最後の「、」、それもなければ指定文字数で改行します。"
    )
    form.addRow("読み上げの改行目安", window.tts_line_break_target_spin)

    required_row = QHBoxLayout()
    window.grok_history_required_match_mode_combo = _NoWheelComboBox()
    window.grok_history_required_match_mode_combo.addItem("いずれかを含む", "any")
    window.grok_history_required_match_mode_combo.addItem("すべてを含む", "all")
    required_row.addWidget(
        QLabel("下の語を返答本文に含まない候補を除外（1行に1語・空欄なら無効）")
    )
    required_row.addStretch(1)
    required_row.addWidget(window.grok_history_required_match_mode_combo)
    required_widget = QWidget()
    required_widget.setLayout(required_row)
    form.addRow("必須語ルール", required_widget)

    window.grok_history_response_required_terms_edit = QPlainTextEdit()
    window.grok_history_response_required_terms_edit.setPlaceholderText(
        "例:\n笑う\nうなずく"
    )
    window.grok_history_response_required_terms_edit.setMaximumHeight(100)
    form.addRow("返答の必須語", window.grok_history_response_required_terms_edit)

    window.grok_history_response_preferred_terms_edit = QPlainTextEdit()
    window.grok_history_response_preferred_terms_edit.setPlaceholderText(
        "例:\n照れる\n喜ぶ"
    )
    window.grok_history_response_preferred_terms_edit.setMaximumHeight(100)
    window.grok_history_response_preferred_terms_edit.setToolTip(
        "一致数が多い返答候補を優先します。ランダム選択時も最大一致数の候補群から選びます。"
    )
    form.addRow("返答の優先語", window.grok_history_response_preferred_terms_edit)

    # 期間指定。チェックを外すと制限なし（保存値は空文字になる）。
    window.grok_history_date_from_chk = QCheckBox("この日以降")
    window.grok_history_date_from_edit = QDateEdit()
    window.grok_history_date_from_edit.setCalendarPopup(True)
    window.grok_history_date_from_edit.setDisplayFormat("yyyy-MM-dd")
    window.grok_history_date_from_edit.setDate(QDate.currentDate().addYears(-1))
    window.grok_history_date_from_edit.setEnabled(False)
    window.grok_history_date_from_chk.toggled.connect(
        window.grok_history_date_from_edit.setEnabled
    )

    window.grok_history_date_to_chk = QCheckBox("この日以前")
    window.grok_history_date_to_edit = QDateEdit()
    window.grok_history_date_to_edit.setCalendarPopup(True)
    window.grok_history_date_to_edit.setDisplayFormat("yyyy-MM-dd")
    window.grok_history_date_to_edit.setDate(QDate.currentDate())
    window.grok_history_date_to_edit.setEnabled(False)
    window.grok_history_date_to_chk.toggled.connect(
        window.grok_history_date_to_edit.setEnabled
    )

    date_row = QHBoxLayout()
    date_row.addWidget(window.grok_history_date_from_chk)
    date_row.addWidget(window.grok_history_date_from_edit)
    date_row.addSpacing(12)
    date_row.addWidget(window.grok_history_date_to_chk)
    date_row.addWidget(window.grok_history_date_to_edit)
    date_row.addStretch(1)
    date_widget = QWidget()
    date_widget.setLayout(date_row)
    date_widget.setToolTip(
        "会話の日付で候補を絞ります。指定した日は範囲に含みます。\n"
        "絞り込みはサーバ側の検索時に行うため、期間内から候補数ぶんが取得されます。"
    )
    form.addRow("期間", date_widget)

    note = QLabel(
        "候補処理: ベクター上位候補取得 → 最低スコア/必須語で除外 → 優先語一致数で並べ替え → 最上位またはランダム選択。\n"
        "改行処理: 目安文字数内の最後の「。/！/!/ハート」→ 最後の「、」→ 指定文字数の順で分割。"
    )
    note.setWordWrap(True)
    layout.addWidget(note)

    _build_server_section(window, layout)
    _build_update_section(window, layout)
    layout.addStretch(1)


def _build_server_section(window, layout) -> None:
    """検索API(既定8877)とOllamaの起動設定。GUIと同じ寿命で自動起動する。"""
    box = QGroupBox("検索サーバ / Ollama")
    form = QFormLayout(box)

    window.grok_history_autostart_chk = QCheckBox("起動時に検索APIを自動で立ち上げる")
    window.grok_history_autostart_chk.setToolTip(
        "このアプリを閉じるとサーバも停止します。\n"
        "既に同じポートで動いているものがあれば、それをそのまま使います。"
    )
    form.addRow("検索API", window.grok_history_autostart_chk)

    window.grok_history_api_port_spin = QSpinBox()
    window.grok_history_api_port_spin.setRange(1, 65535)
    window.grok_history_api_port_spin.setValue(8877)
    window.grok_history_api_port_spin.setToolTip(
        "検索APIの待ち受けポート。上の「検索API」欄のURLと合わせてください。"
    )
    form.addRow("検索APIポート", window.grok_history_api_port_spin)

    window.grok_history_ollama_autostart_chk = QCheckBox("必要ならOllamaも自動で起動する")
    window.grok_history_ollama_autostart_chk.setToolTip(
        "埋め込みに使います。既に起動している場合は何もしません（他で使っていても停止しません）。\n"
        "自分で起動した場合だけ、このアプリの終了時に一緒に停止します。"
    )
    form.addRow("Ollama", window.grok_history_ollama_autostart_chk)

    window.grok_history_ollama_endpoint_edit = QLineEdit()
    window.grok_history_ollama_endpoint_edit.setPlaceholderText("http://127.0.0.1:11434")
    window.grok_history_ollama_endpoint_edit.setToolTip(
        "Ollamaの接続先。別マシンを指した場合、こちらからは起動できないので\n"
        "そのマシンで起動しておく必要があります。"
    )
    form.addRow("Ollama接続先", window.grok_history_ollama_endpoint_edit)

    exe_row = QHBoxLayout()
    window.grok_history_ollama_exe_edit = QLineEdit()
    window.grok_history_ollama_exe_edit.setPlaceholderText(
        "空欄なら自動で探します（%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe など）"
    )
    browse_btn = QPushButton("参照...")
    exe_row.addWidget(window.grok_history_ollama_exe_edit, 1)
    exe_row.addWidget(browse_btn)
    exe_widget = QWidget()
    exe_widget.setLayout(exe_row)
    form.addRow("ollama.exe", exe_widget)

    def _browse_ollama() -> None:
        path, _ = QFileDialog.getOpenFileName(
            window, "ollama.exe を選択", "", "実行ファイル (*.exe);;すべて (*.*)"
        )
        if path:
            window.grok_history_ollama_exe_edit.setText(path)

    browse_btn.clicked.connect(_browse_ollama)

    window.grok_history_ollama_model_edit = QLineEdit()
    window.grok_history_ollama_model_edit.setPlaceholderText("bge-m3:latest")
    window.grok_history_ollama_model_edit.setToolTip(
        "埋め込みモデル。索引を作った時と同じものでなければ検索結果が壊れます。\n"
        "変更する場合は履歴の再ベクター化と索引の作り直しが必要です。"
    )
    form.addRow("埋め込みモデル", window.grok_history_ollama_model_edit)

    layout.addWidget(box)


def _build_update_section(window, layout) -> None:
    """Grok履歴の取り込み(差分)を手動実行するボタンと進捗表示。"""
    box = QGroupBox("Grok履歴の更新")
    box_layout = QVBoxLayout(box)

    desc = QLabel(
        "grok_history\\incoming\\ に置いたエクスポートZIPを取り込み、"
        "新規分だけベクター化して索引まで更新します。\n"
        "ZIPは「今までの全部」が入っていて構いません（取り込み済みは重複しません）。"
        "埋め込み処理には Ollama の起動が必要です。"
    )
    desc.setWordWrap(True)
    box_layout.addWidget(desc)

    row = QHBoxLayout()
    window.grok_history_update_btn = QPushButton("Grok履歴を更新")
    window.grok_history_update_btn.setToolTip(
        "取り込み → チャンク生成(新規のみ) → 埋め込み(未処理のみ) → FAISS索引再構築 を順に実行します。"
    )
    window.grok_history_update_status_label = QLabel("待機中")
    row.addWidget(window.grok_history_update_btn)
    row.addWidget(window.grok_history_update_status_label, 1)
    box_layout.addLayout(row)

    window.grok_history_update_log = QPlainTextEdit()
    window.grok_history_update_log.setReadOnly(True)
    window.grok_history_update_log.setMaximumHeight(150)
    window.grok_history_update_log.setPlaceholderText("実行ログがここに出ます。")
    box_layout.addWidget(window.grok_history_update_log)

    layout.addWidget(box)

    window.grok_history_update_btn.clicked.connect(
        lambda: _start_grok_history_update(window)
    )


def _start_grok_history_update(window) -> None:
    """取り込みを別スレッドで走らせる。埋め込みは分単位で掛かるためGUIを止めない。"""
    if getattr(window, "_grok_history_update_thread", None) is not None:
        return

    from PyQt6.QtCore import QThread

    window.grok_history_update_btn.setEnabled(False)
    window.grok_history_update_status_label.setText("実行中...")
    window.grok_history_update_log.clear()

    thread = QThread()
    worker = _GrokHistoryUpdateWorker()
    worker.moveToThread(thread)

    worker.log.connect(window.grok_history_update_log.appendPlainText)

    def _on_finished(ok: bool, summary: str) -> None:
        window.grok_history_update_status_label.setText(
            ("完了: " if ok else "失敗: ") + summary
        )
        window.grok_history_update_log.appendPlainText(
            ("[OK] " if ok else "[NG] ") + summary
        )
        window.grok_history_update_btn.setEnabled(True)
        thread.quit()

    def _on_thread_done() -> None:
        window._grok_history_update_thread = None
        window._grok_history_update_worker = None

    worker.finished.connect(_on_finished)
    thread.started.connect(worker.run)
    thread.finished.connect(_on_thread_done)

    # ローカル変数のままだとGCで消えるのでwindowへ保持する。
    window._grok_history_update_thread = thread
    window._grok_history_update_worker = worker
    thread.start()


class _GrokHistoryUpdateWorker(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def run(self) -> None:
        try:
            from services import grok_history_ingest

            result = grok_history_ingest.ingest(on_output=self.log.emit)
            if not result.ok:
                self.finished.emit(False, result.error)
                return
            self.finished.emit(
                True,
                f"ZIP {result.imported_zips}件 / 新規チャンク {result.new_chunks} / "
                f"埋め込み {result.embedded}",
            )
        except Exception as exc:  # noqa: BLE001 - 失敗理由をそのまま見せる
            self.finished.emit(False, f"{type(exc).__name__}: {exc}")
