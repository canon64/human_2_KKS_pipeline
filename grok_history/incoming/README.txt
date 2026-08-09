Grokのエクスポート ZIP をこのフォルダに置いて、
リポジトリ直下の update_grok_history.bat を実行する。

- ZIPは「今までの全部」が入っていて構わない。
  取り込み済みの会話は差分判定で素通りするので、重複しない。
- 埋め込み処理で Ollama が必要（未起動だと3段目で失敗する）。
- 取り込み済みZIPは runtime/data/grok_export_index/sources/ へ退避される。
