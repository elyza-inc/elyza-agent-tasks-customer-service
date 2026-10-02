# evaluation/ の構成

| ディレクトリ | 役割 |
|---|---|
| `cli/` | エントリポイント。`run_eval.py`(実行+採点、`score_only` で保存済み record の再採点) |
| `engine/` | 会話実行系。`package_runtime.py`(シナリオ世界・ツール・評価対象モデルとの往復)、`package_adapter.py`(タスクと正解から採点入力への変換) |
| `core/` | 入出力と ASR クライアントの共通部品 |
| `contracts/` | データ契約。指標台帳(`metric_inventory.py`)、実行結果や受け渡しのスキーマ検証 |
| `scoring/` | 採点。`package_scoring.py`(record 1件の採点)、`scoring_runner.py`、各指標の実装 |
| `observers/` | ルールベース指標の観測器(日本語・敬語の `japanese_parse_adapter.py` など) |
| `llm/` | LLM クライアントと LLM ジャッジの呼び出し |
| `audio/` | 音声モード。realtime トランスポート、音声指標(`audio_value_metrics.py` は M20/M22/M25/M26、`audio_signal_metrics.py` は M21/M23)、TTS/ASR まわり |
| `aggregate/` | 集計(`summarize_runs.py`) |
| `prompt_templates/` | LLM ジャッジのプロンプト(M05 の同意判定、M15/M16/M19 の会話ログ判定) |

実行の流れは `cli/run_eval.py` → `engine/package_runtime.py`(会話実行)→
`scoring/package_scoring.py`(採点)→ `aggregate/summarize_runs.py`(集計)です。
設定(ジャッジ設定、ASR プロファイル、指標台帳 JSON ほか)はリポジトリ直下の `configs/` にあります。
実行手順は docs/RUN.md を、指標の定義は docs/METRICS.md を参照してください。
