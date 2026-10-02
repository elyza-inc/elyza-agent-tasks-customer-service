"""Regression checks for M11 sub-facet aggregate table."""

import json
import tempfile
import unittest
from pathlib import Path

from elyza_agent_tasks_customer_service.evaluation.aggregate.summarize_runs import (
    audio_table,
    entity_score,
    load_m11_subfacets,
    m11_subfacet_table,
)


class SummarizeRunsTest(unittest.TestCase):
    def test_entity_score_counts_contract_invalid_as_zero(self) -> None:
        self.assertEqual(
            0.0,
            entity_score({"metric_id": "M01", "status": "contract_invalid"}),
        )

    def test_entity_score_uses_m15_and_m19_rates(self) -> None:
        for metric_id in ("M15", "M19"):
            with self.subTest(metric_id=metric_id):
                self.assertEqual(
                    0.25,
                    entity_score(
                        {
                            "metric_id": metric_id,
                            "status": "fail",
                            "value": {"score": 0.25},
                        }
                    ),
                )

    def test_entity_score_uses_m09_inverse_open_count(self) -> None:
        self.assertEqual(
            0.5,
            entity_score(
                {
                    "metric_id": "M09",
                    "status": "measured",
                    "value": {"score_inverse_k": 0.5, "applicable_pass_rate": 1.0},
                }
            ),
        )

    def test_entity_score_falls_back_to_binary_without_rate(self) -> None:
        for metric_id in ("M15", "M19"):
            with self.subTest(metric_id=metric_id):
                self.assertEqual(
                    1.0,
                    entity_score(
                        {"metric_id": metric_id, "status": "pass", "value": {}}
                    ),
                )
                self.assertEqual(
                    0.0,
                    entity_score({"metric_id": metric_id, "status": "fail"}),
                )

    def test_load_m11_subfacets_keeps_record_without_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scored = root / "scored" / "baseline" / "run_001"
            scored.mkdir(parents=True)
            (scored / "score.json").write_text(json.dumps({
                "scenario_id": "scored",
                "metric_results": [{
                    "metric_id": "M11",
                    "diagnostics": {"source_result": {"details": {"sub_facets": {
                        "required_success": {"status": "pass"},
                    }}}},
                }],
            }), encoding="utf-8")
            failed = root / "failed" / "baseline" / "run_001"
            failed.mkdir(parents=True)
            (failed / "record.json").write_text("{}", encoding="utf-8")

            subfacets = load_m11_subfacets([root])

        self.assertEqual("pass", subfacets["baseline"]["scored"]["required_success"])
        self.assertEqual({}, subfacets["baseline"]["failed"])

    def test_m11_subfacet_table_counts_missing_scores_as_zero(self) -> None:
        base = {
            "scored": {
                "required_success": "pass",
                "argument_value": "pass",
                "argument_provenance": "pass",
                "dependency_order": "pass",
                "error_recovery": "N/A",
            },
            "failed_run": {},
            "failed_facet": {
                "required_success": "fail",
                "argument_value": "fail",
                "argument_provenance": "fail",
                "dependency_order": "fail",
                "error_recovery": "fail",
            },
        }
        fired = {
            "scored": {
                "required_success": "pass",
                "argument_value": "pass",
                "argument_provenance": "pass",
                "dependency_order": "pass",
                "error_recovery": "N/M",
            }
        }

        table = m11_subfacet_table(base, fired)

        self.assertIn("| 必須操作の成功 | 0.333 | 1.000 | 3/1 |", table)
        self.assertIn("| エラー対処 | 0.000 | 0.000 | 1/1 |", table)

    def test_audio_category_uses_five_rates_and_requires_all_of_them(self) -> None:
        audio = {
            metric_id: {"status": "passed"}
            for metric_id in ("M20", "M21", "M22", "M23", "M25", "M26")
        }
        table = audio_table({"run": {"audio": audio}})
        self.assertIn("| 音声カテゴリ | 1.000 | — | — | — |", table)
        self.assertEqual("音声カテゴリはM21(参考)を除く5指標の平均", table[-1])

        del audio["M26"]
        table = audio_table({"run": {"audio": audio}})
        self.assertIn("| 音声カテゴリ | — | — | — | — |", table)


if __name__ == "__main__":
    unittest.main()
