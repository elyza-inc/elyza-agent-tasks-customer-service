# 出力の例

[docs/RUN.md](../docs/RUN.md) の1シナリオの確認(テキストと音声)を実行して得た出力です。API を呼ばずに出力の形式を確かめられます。
顧客役と LLM ジャッジはどちらも gpt-5.6-luna、シナリオは htl-001、条件は baseline です。

```text
text-gpt-5.6-luna/htl-001/record.json                          テキストモードの実行記録(会話、ツール呼び出し、実行の統計)
text-gpt-5.6-luna/htl-001/score.json                           同じシナリオの採点結果(指標ごとの status と根拠)
audio-gpt-realtime-2.1-mini/htl-001/record.json                音声モード(realtime)の実行記録
audio-gpt-realtime-2.1-mini/htl-001/score.json                 同じシナリオの採点結果(音声指標を含む)
audio-gpt-realtime-2.1-mini/htl-001/audio_metric_results.json  音声指標の出力
```

score.json の `metric_results[].status` は、pass、fail、measured(率で採点)、N/A(対象外)、N/M(測定できない。集計では0点)のどれかです。指標のIDと名前の対応は、[docs/METRICS.md の指標の一覧](../docs/METRICS.md#指標の一覧)を参照してください。どれも1回の実行の例で、README の評価結果の値とは一致しません。
