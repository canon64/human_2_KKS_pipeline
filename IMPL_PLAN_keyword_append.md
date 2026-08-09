# IMPL_PLAN: 特定ワードに反応して追加文言を送る

## 目的

発話に特定の言葉が含まれていたとき、その言葉に紐づく文言を LLM への入力に足す。

例:
- 「スタイル」→ 体位一覧を添える
- 「曲を変えて」→ 曲リストを添える

複数のルールを登録でき、それぞれ独立して有効/無効を切り替えられる。

## 必要材料（調査済み）

| # | 材料 | 結果 |
|---|---|---|
| 1 | 追記の合流点 | `compose_llm_input(text, always_append_text)`（llm_providers.py:46）。`tts_event_cli.py:1314` で LLM 送信直前に1回だけ呼ばれる |
| 2 | 既存の固定追記 | `llm_always_append_text`（単一文字列）。GUI「毎回追加ワード」。**条件なしで毎回足す**もの |
| 3 | 表形式UIの前例 | `filter_table`（QTableWidget 4列: 有効/パターン/種別/登録順）。`_ConversionTableDelegate` で行追加、検索・並べ替えあり |
| 4 | 表の保存形式 | `filter_phrases: list[dict]`（`{"enabled":bool,"pattern":str,"type":str}`）。そのまま JSON に載る |
| 5 | 一致方式の前例 | `partial`（部分一致）/ `exact`（完全一致）/ `regex`（正規表現）の3種。GUIでは「部分一致/完全一致/正規表現」と表示 |
| 6 | 子プロセスへの受け渡し | `pipeline_worker.py` が `tts_event_cli` を引数で起動。**長い文言はコマンドライン引数に載せられない** |
| 7 | 長文の受け渡し前例 | `send_voice_face_event.ps1` は `-JsonFile` でUTF-8ファイル経由。同じ方式が使える |

### 材料#6が設計上の要点

体位一覧・曲リストは長くなる。Windowsのコマンドライン長には上限があり、
引数で渡すと確実に破綻する。**設定ファイルを子プロセス側で直接読む**方式にする。

## 設計

### データ形式

`config.json` に新規キー `llm_keyword_appends` を追加する。

```json
"llm_keyword_appends": [
  {"enabled": true, "pattern": "スタイル", "type": "partial", "append": "体位一覧:\n1. ..."},
  {"enabled": true, "pattern": "曲を変え",  "type": "partial", "append": "曲リスト:\n..."}
]
```

`filter_phrases` と同じ形に `append`（送る文言）を足しただけ。既存の読み書きを流用できる。

### 判定と合流

`compose_llm_input()` を拡張し、ルール一覧を受け取れるようにする。

1. 有効なルールを上から順に判定
2. 一致したルールの `append` を集める（重複は除く）
3. 既存の「毎回追加ワード」と合わせて、本文の後ろに連結

**複数一致は全部足す**。「スタイル」と「曲」が同時に出たら両方の一覧を送る。

### 文言の指定方法（2通り）

`append` の値は、次のどちらでもよい。

1. **文言そのもの** — 短いものはここに直接書く
2. **txtファイルのパス** — `.txt` で終わり、実在すればその中身を読む

パスは**プラグインフォルダ起点の相対パス**でも絶対パスでも可。相対にしておけば
配置場所が変わっても壊れない。読み込みは必ず UTF-8。

ファイルが見つからない場合は、その文字列自体を文言として扱う（黙って空にしない）。

### 受け渡し

`tts_event_cli` に `--llm-keyword-appends-file` を追加し、
`pipeline_worker` が一時JSONを書いて渡す。引数長の制限を受けない。

## 変更ファイル

| ファイル | 変更 |
|---|---|
| `config/models.py` | `llm_keyword_appends: list[dict]` を追加 |
| `gui/main_window.py` | 表UI（有効/ワード/種別/送る文言）。`filter_table` を雛形にする |
| `controllers/settings_controller.py` | 保存・読込 |
| `workers/pipeline_worker.py` | 一時JSONへ書き出して引数で渡す |
| `grok_bridge/tts_event_cli.py` | 引数追加、ファイル読込、`compose_llm_input` へ渡す |
| `grok_bridge/llm_providers.py` | `compose_llm_input` にルール判定を追加 |

## 検証

1. 部分一致・完全一致・正規表現それぞれの判定（単体）
2. 複数一致で両方足されること（単体）
3. 無効行が無視されること（単体）
4. 長文（数千文字）が引数経由でなく届くこと
5. 実機で「スタイル」を含む発話に一覧が添えられること

## 未確定・確認したいこと

- **添える位置**: 本文の後ろに足す想定。前に置きたい場合は要指定
- **LLMに送る文だけか、読み上げにも影響するか**: 既存の `always_append` は
  「LLM/検索に送る文にだけ足す」設計なので、それに合わせる（読み上げには出ない）
- **一覧の中身**: 体位一覧・曲リストの実データは未取得。GUIで入力する前提
