from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
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

    note = QLabel(
        "候補処理: ベクター上位候補取得 → 最低スコア/必須語で除外 → 優先語一致数で並べ替え → 最上位またはランダム選択。\n"
        "改行処理: 目安文字数内の最後の「。/！/!/ハート」→ 最後の「、」→ 指定文字数の順で分割。"
    )
    note.setWordWrap(True)
    layout.addWidget(note)
    layout.addStretch(1)
