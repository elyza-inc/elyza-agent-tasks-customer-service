#!/usr/bin/env python3
"""Summarize run_eval score outputs into the published metric tables.

``--results-root``(repeatable)配下から ``**/run_001/score.json`` を再帰探索し、
変種(baseline / hard)ごとの指標別表、カテゴリ表、走行結果3分類、音声指標表
(``audio_user/audio_metric_results.json`` がある場合)を Markdown で出力する。

集計規則(公開値と同一):
- pass=1 / fail=0 / N/M・contract_invalid=0(分母に算入)/ N/A=除外
- measured の数値: M09=score_inverse_k、M11=applicable_pass_rate、M16=fired_item_pass_rate
- M14 は ticket_fidelity_deterministic、M15/M19 は score があれば率を使う
- M04/M19 は hard 単独(baseline は構造的N/A)
- 対応差: 両変種とも採点値がある実体のみ、bootstrap 95%CI(seed 20260813)
- カテゴリの公開値: タスク遂行・SOP・ツール・応対記録は両変種とも満点(>=0.999)の実体の割合、
  対話品質は両変種の平均、義務・耐性と困難応対は hard の平均
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from elyza_agent_tasks_customer_service.evaluation.contracts.variants import normalize_variant

METRICS = ["M01", "M04", "M05", "M09", "M11", "M14", "M15", "M16", "M17", "M19", "M24"]
HARD_ONLY = {"M04", "M19"}
NUMERIC_KEY = {"M09": "score_inverse_k", "M11": "applicable_pass_rate",
               "M16": "fired_item_pass_rate"}
AUDIO_METRICS = ["M20", "M22", "M25", "M26", "M21", "M23"]
AUDIO_CATEGORY_METRICS = ["M20", "M22", "M23", "M25", "M26"]
M11_SUBFACETS = [
    ("required_success", "必須操作の成功"),
    ("argument_value", "引数の値"),
    ("argument_provenance", "引数の出所"),
    ("dependency_order", "操作の順序"),
    ("error_recovery", "エラー対処"),
]
NOT_APPLICABLE_STATUSES = {"N/A", "not_applicable"}
CATEGORIES = [
    ("タスク遂行", ["M01"], False), ("SOP", ["M09"], False), ("ツール", ["M11"], False),
    ("応対記録", ["M14", "M15"], False), ("対話品質", ["M16", "M17", "M24"], False),
    ("義務・耐性", ["M04", "M05"], True), ("困難応対", ["M19"], True),
]
BOOTSTRAP_SEED = 20260813
MEAN_PUBLISHED_CATEGORIES = {"対話品質"}
DOMAIN_BY_PREFIX = {"htl": "hotel", "tel": "telecom", "par": "parcel", "ecf": "ec_flea"}
RUN_FAILED_PREFIX = "run_failed:"
LIMIT_FAILURE_REASONS = ("max_turns", "max_tool_rounds")
RUN_OUTCOMES = ("終話到達", "上限打ち切り", "測定エラー")


def entity_score(metric: dict) -> float | None:
    status = metric["status"]
    if status == "N/A":
        return None
    if metric["metric_id"] == "M14" and status in ("pass", "fail"):
        value = (metric.get("value") or {}).get("ticket_fidelity_deterministic")
        return float(value) if value is not None else (1.0 if status == "pass" else 0.0)
    if metric["metric_id"] in ("M15", "M19") and status in ("pass", "fail"):
        value = (metric.get("value") or {}).get("score")
        return float(value) if value is not None else (1.0 if status == "pass" else 0.0)
    if status == "pass":
        return 1.0
    if status in ("fail", "N/M", "contract_invalid"):
        return 0.0
    if status == "measured":
        value = (metric.get("value") or {}).get(NUMERIC_KEY[metric["metric_id"]])
        return float(value) if value is not None else 0.0
    raise ValueError(f"unknown metric status: {status}")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_outcome(metric_results: list[dict]) -> str:
    """Classify a score.json ``metric_results`` list by its run_failed reasons.

    A reason beginning with ``run_failed:`` is required to mark a failed run.
    Such a reason containing ``max_turns`` or ``max_tool_rounds`` is a limit
    cutoff; every other run failure is a measurement error.
    """

    reasons = [row.get("reason", "") for row in metric_results]
    failures = [reason for reason in reasons if isinstance(reason, str)
                and reason.startswith(RUN_FAILED_PREFIX)]
    if not failures:
        return "終話到達"
    if any(limit in reason for reason in failures for limit in LIMIT_FAILURE_REASONS):
        return "上限打ち切り"
    return "測定エラー"


def boot_ci(diffs: list[float], n: int = 10000, seed: int = BOOTSTRAP_SEED):
    rng = random.Random(seed)
    if not diffs:
        return (None, None)
    means = sorted(mean([rng.choice(diffs) for _ in diffs]) for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def load_runs(roots: list[Path]) -> dict[str, dict]:
    """Load score, event-log, and audio JSON files into the summary shape.

    ``score.json`` must contain ``scenario_id`` and ``metric_results``; optional
    ``event_log.json`` is an event-row array and ``audio_metric_results.json``
    a mapping. Duplicate scenario IDs within a variant raise ``ValueError``.
    """

    variants: dict[str, dict] = {}
    for root in roots:
        run_dirs = {p.parent for p in root.rglob("score.json")}
        run_dirs |= {p.parents[1] for p in root.rglob("audio_metric_results.json")}
        for run_dir in sorted(d for d in run_dirs if d.name == "run_001"):
            variant = normalize_variant(run_dir.parent.name)
            scenario_id = run_dir.parents[1].name
            entry = {"metrics": {}, "outcome": "終話到達", "audio": None,
                     "customer_turns": None, "tool_calls": None}
            score_path = run_dir / "score.json"
            if score_path.exists():
                score = json.loads(score_path.read_text(encoding="utf-8"))
                scenario_id = score["scenario_id"]
                entry["outcome"] = run_outcome(score["metric_results"])
                entry["metrics"] = {m["metric_id"]: (entity_score(m), m["status"])
                                    for m in score["metric_results"]}
            event_path = run_dir / "event_log.json"
            if event_path.exists():
                events = json.loads(event_path.read_text(encoding="utf-8"))
                entry["customer_turns"] = sum(
                    1 for e in events
                    if e.get("actor") == "user" and e.get("event_type") == "message"
                )
                entry["tool_calls"] = sum(
                    1 for e in events if e.get("event_type") == "tool_call"
                )
            audio_path = run_dir / "audio_user" / "audio_metric_results.json"
            if audio_path.exists():
                entry["audio"] = json.loads(audio_path.read_text(encoding="utf-8"))["metrics"]
            bucket = variants.setdefault(variant, {})
            if scenario_id in bucket:
                raise ValueError(f"duplicate scenario under {variant}: {scenario_id}")
            bucket[scenario_id] = entry
    return variants


def load_m11_subfacets(roots: list[Path]) -> dict[str, dict[str, dict[str, str]]]:
    """Load M11 sub-facet statuses from score or record run directories.

    ``score.json`` may contain an M11 metric at
    ``diagnostics.source_result.details.sub_facets``. ``record.json`` without a
    score is retained as an empty mapping so failed runs count as zero where
    applicable. Duplicate scenario IDs within a variant raise ``ValueError``.
    """

    variants: dict[str, dict[str, dict[str, str]]] = {}
    for root in roots:
        run_dirs = {p.parent for p in root.rglob("score.json")}
        run_dirs |= {p.parent for p in root.rglob("record.json")}
        for run_dir in sorted(d for d in run_dirs if d.name == "run_001"):
            variant = normalize_variant(run_dir.parent.name)
            scenario_id = run_dir.parents[1].name
            statuses: dict[str, str] = {}
            score_path = run_dir / "score.json"
            if score_path.exists():
                score = json.loads(score_path.read_text(encoding="utf-8"))
                scenario_id = score.get("scenario_id", scenario_id)
                m11 = next(
                    (m for m in score.get("metric_results", []) if m.get("metric_id") == "M11"),
                    {},
                )
                facets = ((m11.get("diagnostics") or {}).get("source_result") or {}).get(
                    "details", {}
                ).get("sub_facets", {})
                if isinstance(facets, dict):
                    statuses = {
                        name: facet.get("status")
                        for name, facet in facets.items()
                        if isinstance(facet, dict) and isinstance(facet.get("status"), str)
                    }
            bucket = variants.setdefault(variant, {})
            if scenario_id in bucket:
                raise ValueError(f"duplicate scenario under {variant}: {scenario_id}")
            bucket[scenario_id] = statuses
    return variants


def m11_subfacet_table(base: dict[str, dict[str, str]], hard: dict[str, dict[str, str]]) -> list[str]:
    """Return pass rates for M11 facets, counting unreadable applicable facets as zero."""

    lines = ["| M11サブ項目 | baseline | hard | n(b/h) |", "|---|---|---|---|"]
    for name, label in M11_SUBFACETS:
        def col(data: dict[str, dict[str, str]]) -> list[float]:
            values = []
            for statuses in data.values():
                status = statuses.get(name)
                if status in NOT_APPLICABLE_STATUSES:
                    continue
                if name == "error_recovery" and status is None:
                    continue
                values.append(1.0 if status == "pass" else 0.0)
            return values

        bv = col(base)
        hv = col(hard)
        fmt = lambda values: f"{mean(values):.3f}" if values else "—"
        lines.append(f"| {label} | {fmt(bv)} | {fmt(hv)} | {len(bv)}/{len(hv)} |")
    return lines


def metric_table(base: dict, hard: dict) -> list[str]:
    lines = ["| 指標 | baseline | hard | 対応差 [95%CI] | n(対) | N/A(b/h) | N/M(b/h) |",
             "|---|---|---|---|---|---|---|"]
    for mid in METRICS:
        def col(data):
            vals = [e["metrics"][mid][0] for e in data.values()
                    if mid in e["metrics"] and e["metrics"][mid][0] is not None]
            na = sum(1 for e in data.values() if e["metrics"].get(mid, (None, ""))[1] == "N/A")
            nm = sum(1 for e in data.values() if e["metrics"].get(mid, (None, ""))[1] == "N/M")
            return vals, na, nm
        bv, bna, bnm = col(base)
        hv, hna, hnm = col(hard)
        fmt = lambda x: f"{x:.3f}" if x is not None else "—"
        if mid in HARD_ONLY:
            lines.append(f"| {mid} | — | {fmt(mean(hv))} | —(構造的N/A) | {len(hv)} | {bna}/{hna} | {bnm}/{hnm} |")
            continue
        keys = [k for k in base if k in hard
                and base[k]["metrics"].get(mid, (None,))[0] is not None
                and hard[k]["metrics"].get(mid, (None,))[0] is not None]
        diffs = [hard[k]["metrics"][mid][0] - base[k]["metrics"][mid][0] for k in keys]
        lo, hi = boot_ci(diffs)
        ds = f"{mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]" if diffs else "—"
        lines.append(f"| {mid} | {fmt(mean(bv))} | {fmt(mean(hv))} | {ds} | {len(keys)} | {bna}/{hna} | {bnm}/{hnm} |")
    return lines


def category_table(base: dict, hard: dict) -> list[str]:
    """Category means per variant, plus the published value.

    The published value for タスク遂行, SOP, ツール and 応対記録 is the share of
    scenarios whose category score is full (>= 0.999) in both baseline and hard;
    対話品質 is the average of the baseline and hard means; the hard-only
    categories use the hard mean.
    """

    lines = ["| カテゴリ | baseline | hard | 対応差 [95%CI] | n(対) | 公開値 |", "|---|---|---|---|---|---|"]
    for name, mids, hard_only in CATEGORIES:
        def cat(entry):
            vals = [entry["metrics"][m][0] for m in mids
                    if entry["metrics"].get(m, (None,))[0] is not None]
            return mean(vals)
        bv = [cat(e) for e in base.values() if cat(e) is not None]
        hv = [cat(e) for e in hard.values() if cat(e) is not None]
        fmt = lambda x: f"{x:.3f}" if x is not None else "—"
        if hard_only:
            lines.append(f"| {name} | — | {fmt(mean(hv))} | —(誘発時のみ) | {len(hv)} | {fmt(mean(hv))} |")
            continue
        keys = [k for k in base if k in hard and cat(base[k]) is not None and cat(hard[k]) is not None]
        diffs = [cat(hard[k]) - cat(base[k]) for k in keys]
        lo, hi = boot_ci(diffs)
        ds = f"{mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]" if diffs else "—"
        if name in MEAN_PUBLISHED_CATEGORIES:
            # Average the two variant means as printed (3 decimals), so the value matches the table.
            means = [round(x, 3) for x in (mean(bv), mean(hv)) if x is not None]
            published = round(sum(means) / len(means), 3) if means else None
        else:
            published = mean([1.0 if cat(base[k]) >= 0.999 and cat(hard[k]) >= 0.999 else 0.0 for k in keys])
        lines.append(f"| {name} | {fmt(mean(bv))} | {fmt(mean(hv))} | {ds} | {len(keys)} | {fmt(published)} |")
    return lines


def audio_table(data: dict) -> list[str] | None:
    per = {m: {"p": 0, "f": 0, "nm": 0, "na": 0} for m in AUDIO_METRICS}
    obs_num = obs_den = 0
    found = False
    for entry in data.values():
        metrics = entry["audio"]
        if not metrics:
            continue
        found = True
        for mid in AUDIO_METRICS:
            value = metrics.get(mid)
            if not value:
                continue
            units = value.get("units") or []
            statuses = [u.get("status") for u in units] if units else [value.get("status")]
            for status in statuses:
                key = {"passed": "p", "failed": "f", "N/M": "nm", "N/A": "na"}.get(status)
                if key:
                    per[mid][key] += 1
            if mid == "M25":
                obs = value.get("observation") or {}
                target_values = obs.get("target_value_count") or 0
                if target_values:
                    obs_num += min(obs.get("opportunity_count", 0), target_values)
                    obs_den += target_values
    if not found:
        return None
    lines = ["| 音声指標 | 率 | n | N/A | N/M |", "|---|---|---|---|---|"]
    rates = {}
    for mid in AUDIO_METRICS:
        c = per[mid]
        den = c["p"] + c["f"] + c["nm"]
        rates[mid] = c["p"] / den if den else None
        rate = f"{rates[mid]:.3f}" if rates[mid] is not None else "—"
        lines.append(f"| {mid} | {rate} | {den} | {c['na']} | {c['nm']} |")
    if obs_den:
        lines.append(f"| M25観測率 | {obs_num / obs_den:.3f} | {obs_den} | 0 | 0 |")
    category_rate = mean([rates[mid] for mid in AUDIO_CATEGORY_METRICS]) if all(
        rates[mid] is not None for mid in AUDIO_CATEGORY_METRICS
    ) else None
    if category_rate is None:
        lines.append("| 音声カテゴリ | — | — | — | — |")
    else:
        lines.append(f"| 音声カテゴリ | {category_rate:.3f} | — | — | — |")
    lines.append("音声カテゴリはM21(参考)を除く5指標の平均")
    return lines


def domain_table(hard: dict) -> list[str]:
    domains = sorted({DOMAIN_BY_PREFIX.get(sid.split("-")[0], sid.split("-")[0]) for sid in hard},
                     key=lambda d: list(DOMAIN_BY_PREFIX.values()).index(d) if d in DOMAIN_BY_PREFIX.values() else 99)
    lines = ["| ドメイン | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for domain in domains:
        cells = []
        for mid in METRICS:
            vals = [e["metrics"][mid][0] for sid, e in hard.items()
                    if DOMAIN_BY_PREFIX.get(sid.split("-")[0], sid.split("-")[0]) == domain
                    and e["metrics"].get(mid, (None,))[0] is not None]
            cells.append(f"{mean(vals):.2f}" if vals else "—")
        lines.append(f"| {domain} | " + " | ".join(cells) + " |")
    return lines


def side_table(variants: dict) -> list[str]:
    """副次情報。指標の得点ではなく、実行状況を示す補助値。

    終話まで到達した実行だけを対象に、顧客の発話ターン数と
    ツール呼び出し数(手順書検索を含む)の平均を出す。
    """

    lines = ["| 変種 | 終話到達 | 顧客ターン数 | ツール呼び出し数 |",
             "|---|---|---|---|"]
    for variant in sorted(variants):
        entries = list(variants[variant].values())
        done = [e for e in entries if e["outcome"] == "終話到達"]
        turns = [e["customer_turns"] for e in done if e["customer_turns"] is not None]
        calls = [e["tool_calls"] for e in done if e["tool_calls"] is not None]
        fmt = lambda xs: f"{sum(xs) / len(xs):.1f}" if xs else "—"
        lines.append(f"| {variant} | {len(done)}/{len(entries)} | {fmt(turns)} | {fmt(calls)} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    variants = load_runs(args.results_root)
    base = variants.get("baseline", {})
    hard = variants.get("hard", {})
    print(f"実体数 baseline={len(base)} hard={len(hard)}")
    outcome_counts = {
        variant: {outcome: sum(1 for entry in entries.values() if entry["outcome"] == outcome)
                  for outcome in RUN_OUTCOMES}
        for variant, entries in variants.items()
    }
    print(
        f"終話到達 b={outcome_counts.get('baseline', {}).get('終話到達', 0)}/{len(base)}"
        f" h={outcome_counts.get('hard', {}).get('終話到達', 0)}/{len(hard)}"
        f" | 上限打ち切り b={outcome_counts.get('baseline', {}).get('上限打ち切り', 0)}"
        f" h={outcome_counts.get('hard', {}).get('上限打ち切り', 0)}"
        f" | 測定エラー b={outcome_counts.get('baseline', {}).get('測定エラー', 0)}"
        f" h={outcome_counts.get('hard', {}).get('測定エラー', 0)}"
    )
    print()
    print("\n".join(metric_table(base, hard)))
    m11_subfacets = load_m11_subfacets(args.results_root)
    print()
    print("\n".join(m11_subfacet_table(
        m11_subfacets.get("baseline", {}), m11_subfacets.get("hard", {})
    )))
    print("値は走行失敗を0点に算入した合格率。エラー対処はエラー注入シナリオのみが対象。")
    print()
    print("\n".join(category_table(base, hard)))
    if hard:
        print("\n### ドメイン別(hard)")
        print("\n".join(domain_table(hard)))
    print("\n### 副次情報(実行状況)")
    print("\n".join(side_table(variants)))
    for variant in sorted(variants):
        table = audio_table(variants[variant])
        if table:
            print(f"\n### 音声指標({variant})")
            print("\n".join(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
