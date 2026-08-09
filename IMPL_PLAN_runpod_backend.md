# IMPL_PLAN: RunPod (Open WebUI + Ollama) をLLMバックエンドに追加

## 目的

頭脳役の選択肢に「RunPod上のOpen WebUI経由のOllama」を追加する。
既存の `grok_browser` / `local_openai` は変更せず残す。

## 必要材料（調査済み）

| # | 材料 | 結果 | 確認方法 |
|---|---|---|---|
| 1 | バックエンド分岐の所在 | `grok_bridge/llm_providers.py` の `normalize_backend()` と `generate_llm_response()` に集約 | ソース読解 |
| 2 | `local_openai` に接続先制限があるか | **無し**。`_is_loopback_http_url()` はGrok履歴検索URLの検証にのみ使用（llm_providers.py:154） | grep で使用箇所を全走査 |
| 3 | RunPod側のOpenAI互換窓口の有無 | **有り**。`{base}/ollama/v1/chat/completions` が HTTP 200 を返し、Cydonia-24B が応答 | 実機に curl 実行 |
| 4 | `/api/chat/completions`（Open WebUI独自） | 動作するがレスポンス形式が独自寄り。採用しない | 同上 |
| 5 | 認証方式 | ログインで得るBearerトークン。`POST /api/v1/auths/signin` に `{email,password}` | 実機確認 |
| 6 | 恒久APIキーの発行可否 | **不可**。`POST /api/v1/auths/api_key` は HTTP 403（`API key creation is not allowed in the environment.`） | 実機確認 |
| 7 | トークン有効期限 | **28日**（`expires_at` を実測） | 実機確認 |
| 8 | 設定項目がGUI→末端まで流れるか | **流れる**。`llm_backend/base_url/model/api_key/system_prompt/temperature/max_tokens/timeout_seconds` は全て GUI → `AppConfig` → `pipeline_worker` → `tts_event_cli` → `LlmRequestConfig` に到達 | grep で経路確認 |
| 9 | `--llm-backend` に選択肢制限があるか | **無し**（`choices=` 未指定、tts_event_cli.py:684）。新しい値をそのまま通せる | ソース読解 |

### 結論
**新規の設定項目・GUI項目・CLI引数の追加は不要。** 既存項目の意味を拡張するだけで実現できる。

## 設計

### 既存項目の再利用

| 既存項目 | RunPod時の意味 |
|---|---|
| `llm_backend` | `runpod_openwebui`（新しい値） |
| `llm_base_url` | Open WebUIのURL、**またはPod IDだけ**（例: `fofvwgs2js8mjy`） |
| `llm_api_key` | `email:password` 形式ならログインして自動取得。それ以外はトークンそのものとして使用 |
| `llm_model` | Ollamaのモデル名（例: `hf.co/TheDrummer/Cydonia-24B-v4.3-GGUF:Q4_K_M`） |

### URL正規化

入力を以下の順で解釈する。

1. `https://` / `http://` で始まる → そのまま採用
2. それ以外（Pod IDと判断） → `https://{id}-8080.proxy.runpod.net` に展開
3. 末尾に `/ollama/v1` が無ければ付与

これにより **Pod を立て直したときはPod IDの貼り替え1箇所で済む**（案B相当）。
フルURLを直接書く運用（案A）も同じ経路で成立する。

### トークン取得とキャッシュ

- `llm_api_key` に `:` が含まれる → `email:password` とみなす
- プロセス内に `(base_url, email)` をキーとしてトークンをキャッシュ
- HTTP 401/403 を受けたら **1回だけ** 再ログインして再送（期限切れ・Pod再作成に自動追従）
- `:` を含まない場合は従来通りそのままBearerに載せる（トークン直指定）

### リクエスト本体

`_generate_openai_compatible_response()` をそのまま流用する。
OpenAI互換なのでボディ組み立て・レスポンス解釈は共通で済む。

## 変更ファイル

| ファイル | 変更内容 | 規模 |
|---|---|---|
| `grok_bridge/llm_providers.py` | `runpod_openwebui` を追加。URL正規化・トークン取得/キャッシュ・401再試行を実装 | 主 |
| `gui/main_window.py` | バックエンド選択肢に1件追加、入力欄のヒント文言を追記 | 小 |
| `grok_bridge/tts_event_cli.py` | `--llm-backend` のヘルプ文言を更新 | 極小 |

`config/models.py` `controllers/settings_controller.py` `workers/pipeline_worker.py` は**変更不要**（材料#8）。

## 非対象

- RunPod APIキーによるPod自動起動（案C）。別タスクとする
- ストリーミング応答。既存の `local_openai` も非対応のため揃える

## 検証

1. `normalize_backend()` の別名解決（単体）
2. URL正規化：Pod ID / フルURL / 末尾スラッシュ有無（単体）
3. `email:password` 判定とトークン直指定の分岐（単体）
4. 401時に再ログインして成功すること（モック）
5. 実機：Pod に対して実際に応答が返ること

## 未確定・リスク

- **Pod停止中はURLごと消える。** 接続失敗時のエラー文言で「Podが停止している可能性」を示す必要がある
- Open WebUI のAPIパスは将来のバージョンで変わりうる。`/ollama/v1` が404になった場合の切り分け手順を README に残すこと
- パスワードが設定ファイルに平文で入る。Pod内部限定の使い捨て資格情報である前提を明記する
