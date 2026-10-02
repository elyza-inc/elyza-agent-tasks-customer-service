# 実行と採点の手順

評価の実行から集計までのコマンドを載せています。コマンドはリポジトリ直下で実行してください。

| やりたいこと | 節 |
|---|---|
| 評価対象モデルごとの接続設定を知る | [評価対象モデルごとの接続設定](#評価対象モデルごとの接続設定) |
| 初めて動かす | [環境を作る](#環境を作る) |
| 実行用パッケージを作る | [実行用パッケージを作る](#実行用パッケージを作る) |
| まず1シナリオで動作を確かめる | [まず1シナリオで確かめる](#まず1シナリオで確かめる) |
| 全シナリオと両条件を評価する | [全シナリオを実行する](#全シナリオを実行する) |
| 音声モデルを評価する | [音声モードを実行して採点する](#音声モードを実行して採点する) |
| 保存済みの記録を採点し直す | [保存済みの記録を採点し直す](#保存済みの記録を採点し直す) |
| 結果を表にまとめる | [集計する](#集計する) |
| 設定キーの意味を調べる | [主な設定キー](#主な設定キー) |

例はすべてOpenAIのendpointを使い、評価対象、顧客役、LLMジャッジのいずれも `gpt-5.6-luna` です。評価対象モデルにはOpenAI互換のchat/completions APIで接続します。gpt-5.6系のように関数ツールを使うときに `/v1/responses` が要るモデルには `"operator_transport":"responses"` を指定してください。

実行前に `OPENAI_API_KEY` を環境変数に設定してください。`.env` は自動では読み込まれません。ファイルから読ませる場合は、JSON設定の `env_path` に `.env` のパスを指定します(雛形は [.env.template](../.env.template))。音声モードには `ffmpeg` も要ります。

## 評価対象モデルごとの接続設定

READMEの評価結果の表に載せたモデルは、次のどれかの方法で接続します。第2引数のendpoint、JSON設定に足すキー、要る環境変数の対応です。

| 接続方法 | 例 | endpoint(第2引数) | JSON設定に足すもの | 要る環境変数 |
|---|---|---|---|---|
| OpenAI(chat/completions) | responsesが要らないモデル | `https://api.openai.com` | なし | `OPENAI_API_KEY` |
| OpenAI(responses) | gpt-5.6-sol、gpt-5.6-luna | `https://api.openai.com` | `"operator_transport":"responses"` | `OPENAI_API_KEY` |
| Anthropic | claude-opus-5、claude-sonnet-5 | `https://api.anthropic.com` | `"operator_api_key_env":"ANTHROPIC_API_KEY"`、`"user_controller_endpoint":"https://api.openai.com"`、`"user_controller_api_key_env":"OPENAI_API_KEY"`、`"operator_prompt_cache":true`(任意) | `ANTHROPIC_API_KEY`、`OPENAI_API_KEY` |
| Google Vertex AI | google/gemini-3.6-flashなど | `https://aiplatform.googleapis.com/v1beta1/projects/<プロジェクト>/locations/global/endpoints/openapi` | `"operator_api_key_env":"GOOGLE_ACCESS_TOKEN"`、`"user_controller_endpoint":"https://api.openai.com"`、`"user_controller_api_key_env":"OPENAI_API_KEY"` | `GOOGLE_ACCESS_TOKEN`、`OPENAI_API_KEY` |
| OpenAI Realtime(音声) | gpt-realtime-2.1、gpt-realtime-2.1-mini | `https://api.openai.com` | `"operator_transport":"realtime"`、`"io_mode":"audio-audio"` | `OPENAI_API_KEY` |
| Gemini Live(音声) | gemini-live-2.5-flash | `https://aiplatform.googleapis.com` | `"operator_transport":"gemini_live"`、`"io_mode":"audio-audio"`、`"operator_api_key_env":"GOOGLE_ACCESS_TOKEN"`、`"user_controller_endpoint":"https://api.openai.com"`、`"user_controller_api_key_env":"OPENAI_API_KEY"`。`api_model` は `projects/<プロジェクト>/locations/<リージョン>/publishers/google/models/gemini-live-2.5-flash` の形で書く | `GOOGLE_ACCESS_TOKEN`、`OPENAI_API_KEY` |
| OpenAI互換サーバ(自前ホスト) | Qwen3-Omniなど | `http://<ホスト>:<ポート>` | `"user_controller_endpoint":"https://api.openai.com"`、`"user_controller_api_key_env":"OPENAI_API_KEY"`。音声で評価するときは `"operator_transport":"chat_audio"`、`"io_mode":"audio-audio"` | `OPENAI_API_KEY`。サーバがキーを求めるときは、その環境変数を `"operator_api_key_env"` で指定 |

顧客役の接続先 `user_controller_endpoint` は、指定しないと第2引数と同じになります。endpointが `https://api.openai.com` でない行(Anthropic、Google Vertex AI、Gemini Live、OpenAI互換サーバ)では、JSON設定に `"user_controller_endpoint":"https://api.openai.com"` も足してください。足さないと、顧客役のリクエストが評価対象モデルの接続先に送られて失敗します。顧客役のモデルは `user_controller_model`(既定は `gpt-5.6-luna`)で変えられます。

LLMジャッジの接続先とモデルは [configs/conversation_log_judge.yaml](../configs/conversation_log_judge.yaml) で決まり、第2引数やJSON設定の影響は受けません。

`gcloud auth application-default print-access-token` で取ったトークンは1時間で切れます。1時間を超える実行では、`gcloud auth application-default login` を済ませたうえで `GOOGLE_ACCESS_TOKEN=ADC` と設定してください。値が `ADC` のときは、シナリオごとにトークンを取り直します。

## 環境を作る

Python 3.11以上の仮想環境に依存を入れます。

```bash
PYTHON="$(command -v python3.12 || command -v python3.11 || command -v python3)"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), sys.version'
"$PYTHON" -m venv .venv
.venv/bin/pip install -e .
test -n "${OPENAI_API_KEY:-}" || echo 'OPENAI_API_KEY is not set' >&2
command -v ffmpeg >/dev/null || echo 'ffmpeg is not found (needed only for audio mode)' >&2
```

最後の2行は確認だけで、足りないものがあれば警告を表示します。`OPENAI_API_KEY` を `.env` から読ませる場合は、1行目の警告は気にしなくてかまいません。

## 実行用パッケージを作る

`tasks`(モデルへの入力)と `solutions`(正解と採点条件)を結合して、実行用パッケージを作ります。次のコマンドは4ドメインすべてを `.run/packages` に出力します。一部のドメインだけ試すときは、ループから要らないドメインを消してください。

```bash
mkdir -p .run/packages
for domain in ec_flea hotel parcel telecom; do
  .venv/bin/python scripts/assemble_packages.py \
    --tasks "data/tasks/$domain" --solutions "data/solutions/$domain" \
    --output .run/packages
done
find .run/packages -maxdepth 1 -name '*.yaml' -type f | wc -l
```

`96` と表示されれば揃っています。

## テキストモードを実行して採点する

### まず1シナリオで確かめる

接続、認証、出力の形式を確かめるときは `scenario_ids` を指定します。`conversation_log_scoring` を指定しているので、`score.json` は実行と同時にできます。

```bash
.venv/bin/python scripts/run_eval.py \
  '{"mode":"chat_tools","scenario_dir":"'"$PWD"'/.run/packages","scenario_ids":["htl-001"],"output_dir":"'"$PWD"'/.run/records/text-smoke","api_model":"gpt-5.6-luna","operator_transport":"responses","variant_id":"baseline","io_mode":"text-text","max_workers":1,"timeout_sec":600,"user_controller_model":"gpt-5.6-luna","conversation_log_scoring":{"config_path":"'"$PWD"'/configs/conversation_log_judge.yaml","cache_dir":"'"$PWD"'/.run/judge-cache"}}' \
  https://api.openai.com
```

### 全シナリオを実行する

全シナリオを流すときは `scenario_ids` を省きます。baselineとhardは別々に実行してください。

評価対象モデルを替えるときは、baselineとhardの両方のコマンドで、`api_model` を同じモデルIDに変えます。あわせて、第2引数のendpointとJSON設定に足すキーを、[評価対象モデルごとの接続設定](#評価対象モデルごとの接続設定)の表に合わせて変えてください。endpointが `https://api.openai.com` でないときは `user_controller_endpoint` も要ります。

```bash
.venv/bin/python scripts/run_eval.py \
  '{"mode":"chat_tools","scenario_dir":"'"$PWD"'/.run/packages","output_dir":"'"$PWD"'/.run/records/text-baseline","api_model":"gpt-5.6-luna","operator_transport":"responses","variant_id":"baseline","io_mode":"text-text","max_workers":6,"timeout_sec":600,"user_controller_model":"gpt-5.6-luna","conversation_log_scoring":{"config_path":"'"$PWD"'/configs/conversation_log_judge.yaml","cache_dir":"'"$PWD"'/.run/judge-cache"}}' \
  https://api.openai.com

.venv/bin/python scripts/run_eval.py \
  '{"mode":"chat_tools","scenario_dir":"'"$PWD"'/.run/packages","output_dir":"'"$PWD"'/.run/records/text-hard","api_model":"gpt-5.6-luna","operator_transport":"responses","variant_id":"hard","io_mode":"text-text","max_workers":6,"timeout_sec":600,"user_controller_model":"gpt-5.6-luna","conversation_log_scoring":{"config_path":"'"$PWD"'/configs/conversation_log_judge.yaml","cache_dir":"'"$PWD"'/.run/judge-cache"}}' \
  https://api.openai.com
```

## 音声モードを実行して採点する

音声を出せるモデルを評価するときに使います。次はOpenAI Realtime APIの `gpt-realtime-2.1-mini` で1シナリオを確かめる例です。全シナリオを流すときは `scenario_ids` を消し、`max_workers` を `4` にします。評価対象には、音声の入出力とfunction callingに対応したモデルを指定してください。

```bash
.venv/bin/python scripts/run_eval.py \
  '{"mode":"chat_tools","scenario_dir":"'"$PWD"'/.run/packages","scenario_ids":["htl-001"],"output_dir":"'"$PWD"'/.run/records/audio-smoke","api_model":"gpt-realtime-2.1-mini","variant_id":"baseline","operator_transport":"realtime","io_mode":"audio-audio","max_workers":1,"timeout_sec":900,"user_controller_model":"gpt-5.6-luna","conversation_log_scoring":{"config_path":"'"$PWD"'/configs/conversation_log_judge.yaml","cache_dir":"'"$PWD"'/.run/judge-cache"}}' \
  https://api.openai.com
```

全シナリオを両条件で評価するときは、テキストと同じく `variant_id` を `baseline` と `hard` にしたコマンドを別々に実行し、`output_dir` も分けます(例: `.run/records/audio-baseline` と `.run/records/audio-hard`)。

音声モードでは、テキストと同じ会話ログの指標に加えて、音声の指標(重要値の保持、義務案内の音声到達、音質、復唱の音声正確性、根拠付き重要値利用率と、参考値の応答の速さ)を同じ実行の中で採点します。指標の定義は [METRICS.md の音声指標](METRICS.md#音声指標音声モードのみ)を参照してください。音声の指標は `score.json` のほか、`<output_dir>/<scenario_id>/<variant>/run_001/audio_user/audio_metric_results.json` にも保存されます。保存済みの音声の記録を `score_only:true`([次の節](#保存済みの記録を採点し直す))で採点し直すと、音声の指標も作り直します。

## 保存済みの記録を採点し直す

モデルを実行し直さずに、保存済みの `record.json` を採点し直すときは `score_only:true` を使います。次のコマンドは、上のテキストの1シナリオの結果を採点し直します。

```bash
.venv/bin/python scripts/run_eval.py \
  '{"mode":"chat_tools","score_only":true,"scenario_dir":"'"$PWD"'/.run/packages","scenario_ids":["htl-001"],"output_dir":"'"$PWD"'/.run/records/text-smoke","variant_id":"baseline","io_mode":"text-text","conversation_log_scoring":{"config_path":"'"$PWD"'/configs/conversation_log_judge.yaml","cache_dir":"'"$PWD"'/.run/judge-cache"}}'
```

`score_only:true` は、各シナリオの `score.json` を上書きします。元の `score.json` を残したまま、1つの `record.json` を採点して別のファイルに結果を出すときは、`package_scoring.py` を使います。`--package` には、そのシナリオの実行用パッケージを指定します。

```bash
.venv/bin/python scripts/package_scoring.py \
  --package "$PWD/.run/packages/htl-001.yaml" \
  --record "$PWD/.run/records/text-smoke/htl-001/baseline/run_001/record.json" \
  --config "$PWD/configs/conversation_log_judge.yaml" \
  --cache-dir "$PWD/.run/judge-cache" \
  --output "$PWD/.run/records/text-smoke/htl-001/baseline/run_001/score-rescored.json" \
  --env-file /dev/null
```

`package_scoring.py` は `run_eval.py` と違い、`--env-file` を省くとリポジトリ直下の `.env` を自動で読み込みます。上の例の `--env-file /dev/null` は、`.env` を読まずに環境変数のキーだけを使う指定です。`.env` のキーを使うときは、この行を消すか `--env-file .env` にしてください。どちらの場合も、設定済みの環境変数は上書きしません。

## 集計する

`--results-root` はいくつでも渡せます。次はテキストのbaselineとhardの結果をまとめ、指標別、カテゴリ別、ドメイン別の表を標準出力に出します。

```bash
.venv/bin/python scripts/summarize_runs.py \
  --results-root "$PWD/.run/records/text-baseline" \
  --results-root "$PWD/.run/records/text-hard"
```

1回の呼び出しに渡せるのは、同じモデルの同じモードの結果だけです。同じシナリオと条件の結果が2つあるとエラーで止まるので、音声の結果は別の呼び出しで集計してください。

`--results-root` の下にある `<scenario_id>/<variant>/run_001/score.json` をたどって読みます。baselineとhardの両方を渡すと、同じシナリオどうしの差も出ます。

カテゴリ表の「公開値」の列は、READMEの評価結果の表と同じ値です。baselineとhardの両方を渡したときに出ます。

| カテゴリ | 公開値 |
|---|---|
| タスク遂行、SOP、ツール、応対記録 | baselineとhardの両方で満点だったシナリオの割合 |
| 対話品質 | baselineの平均とhardの平均を足して2で割った値 |
| 義務・耐性、困難応対 | hardの平均 |

## 主な設定キー

`run_eval.py` の第1引数はJSON設定、第2引数は評価対象モデルのendpointです。主なキーは次のとおりです。READMEの評価結果は、上の例で指定したキー以外を既定値のまま実行したものです。

| キー | 既定値 | 意味 |
|---|---|---|
| `mode` | (必須) | `chat_tools` を指定 |
| `scenario_dir` | (必須) | 結合済みパッケージのディレクトリ(絶対パス) |
| `scenario_ids` | 全件 | 実行するシナリオIDの配列 |
| `output_dir` | (必須) | 記録と採点結果の出力先(絶対パス) |
| `api_model` | (必須) | 評価対象モデルのID |
| `variant_id` | (必須) | `baseline`(揺さぶりなし)か `hard`(揺さぶりあり) |
| `io_mode` | `text-text` | 音声モードは `audio-audio` |
| `operator_transport` | `chat` | `chat`、`responses`、`realtime`、`gemini_live`、`chat_audio` のどれか |
| `operator_prompt_cache` | `false` | AnthropicとVertexのネイティブAPIでプロンプトキャッシュを使う |
| `user_controller_model` | `gpt-5.6-luna` | 顧客役のモデル |
| `user_controller_endpoint` | 第2引数と同じ | 顧客役の接続先 |
| `customer_tts_model` | `gpt-4o-mini-tts-2025-12-15` | 顧客の声を合成するモデル(OpenAIの /v1/audio/speech)。音声モードだけ |
| `operator_api_key_env` | endpointがapi.openai.comなら `OPENAI_API_KEY` | 評価対象モデルのキーを読む環境変数の名前 |
| `user_controller_api_key_env` | 顧客役の接続先がapi.openai.comなら `OPENAI_API_KEY` | 顧客役のキーを読む環境変数の名前 |
| `env_path` | なし | 読み込む `.env` のパス。設定済みの環境変数は上書きしない |
| `max_workers` | 6 | シナリオを並列に流す数(上限6) |
| `timeout_sec` | 600 | 1リクエストのタイムアウト(秒)。音声では900を使う |
| `max_turns` | 28 | 顧客とのやりとりの上限 |
| `max_tool_rounds` | 40 | ツールを呼ぶ応答の回数の上限。会話全体で数え、1つの応答で複数のツールを呼んでも1回 |
| `sop_search_top_k` | 4 | 業務手順書(SOP)の検索で返す件数 |
| `temperature` | 0.0 | 音声モードの評価対象モデルのtemperature。テキストモードでは、OpenAI互換サーバとVertex AIには0.0を送り、OpenAIのgpt-5系とAnthropicには送らない(APIの既定値になる) |
| `conversation_log_scoring.config_path` | なし(指定を推奨) | LLMジャッジの設定。指定すると実行と同時に採点する |
| `conversation_log_scoring.cache_dir` | なし | ジャッジの応答のキャッシュ。採点し直したときに同じ結果を出すのに使う |
| `limit_scenarios` | なし | 先頭のN件だけ実行する(動作確認用) |
| `operator_reasoning` | なし | `operator_transport` が `responses` のときに、評価対象モデルへ渡す `reasoning` の設定 |
| `request_extra_body` | なし | 評価対象モデルへのリクエストに足すフィールド |
| `voice` | `Puck` | Gemini Liveの評価対象モデルの声 |
| `post_call_model` | `google/gemini-3.6-flash` | Gemini Liveで通話後の応対記録を書くモデル。Live用のモデルは通話後の文字の処理に対応しないため、別のモデルを使う |

ツールの引数がスキーマに合わないときと、応対記録が欠けたときは、ハーネスがモデルに直させます(それぞれ最大2回)。コンテキストの長さを超えたときは送り直します。回数と内容は `record.json` の `schema_repair_count`、`ticket_repair_count`、`context_length_retry` に残ります。

## 所要時間の目安(実測)

1シナリオの平均の実行時間は、テキストが61秒(中央値59秒)、音声が112秒(中央値92秒)です。96シナリオを2条件で流すと、テキスト(6並列)で約35分、音声(4並列)で約1.5時間かかります。会話は平均でテキストが19往復、音声が13往復です。APIの費用は、評価対象モデル、顧客役(gpt-5.6-luna)、LLMジャッジの価格で決まります。まず `limit_scenarios` か1ドメインで小さく試すことを勧めます。

## ファイル名の決まり

tasks/とsolutions/のファイル名は `<scenario_id>.json`、実行用パッケージのファイル名は `<scenario_id>.yaml` です。

## 出力の例

実行済みの `record.json` と `score.json` を [examples](../examples)に入れています。APIを呼ばずに出力の形式を確かめられます。
