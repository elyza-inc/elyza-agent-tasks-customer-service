"""Regression checks for M11 score-side argument reattribution."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from elyza_agent_tasks_customer_service.evaluation.contracts.type_contracts import (
    provenance_text_contains_declared_spoken,
    provenance_text_contains_typed_value,
)
from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import (
    convert_package,
    load_package,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.conversation_log_metrics import (
    _attempts,
    _recalculate_argument_provenance,
    _score_m11,
    score_conversation_log_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
HTL_001_PACKAGE_PATH = ROOT / ".run/packages/htl-001.yaml"
HTL_001_RECORD_PATH = (
    ROOT
    / ".run/records-claude-opus-5-baseline/htl-001/baseline/run_001/record.json"
)


class ArgumentProvenanceReattributionTest(unittest.TestCase):
    def test_embedded_date_and_spoken_contact_phone_match(self) -> None:
        self.assertTrue(
            provenance_text_contains_typed_value(
                "オオタユイ、1984年5月5日です。",
                "1984-05-05",
                [("birthdate", "date")],
            )
        )
        self.assertTrue(
            provenance_text_contains_typed_value(
                "ゼロキュウゼロの、ゼロゼロゼロゼロの、ゼロゼロサンゴです",
                "090-0000-0035",
                [("contact_phone", "string")],
            )
        )

    def test_japanese_clock_text_matches_hh_mm(self) -> None:
        cases = (
            ("到着は22時です", "22:00"),
            ("到着は二十二時です", "22:00"),
            ("到着はニジュウニ時です", "22:00"),
            ("夜の十時で登録をお願いします", "22:00"),
            ("到着は22時30分です", "22:30"),
            ("到着は二十二時三十分です", "22:30"),
        )
        for candidate, expected in cases:
            with self.subTest(candidate=candidate, expected=expected):
                self.assertTrue(
                    provenance_text_contains_typed_value(
                        candidate,
                        expected,
                        [("planned_checkin_time", "time")],
                    )
                )
        self.assertFalse(
            provenance_text_contains_typed_value(
                "到着は21時です",
                "22:00",
                [("planned_checkin_time", "time")],
            )
        )

    def test_iso_date_datetime_and_typed_numbers_match_spoken_text(self) -> None:
        cases = (
            ("十一月二日でお願いします", "2026-11-02", [("redelivery_date", "date")]),
            (
                "2026年9月1日14時を希望します",
                "2026-09-01 14:00",
                [("new_departure_time", "string")],
            ),
            ("0013です", 13, [("account_last4", "integer")]),
            ("五十人くらいです", 50, [("capacity", "integer")]),
        )
        for candidate, expected, contracts in cases:
            with self.subTest(candidate=candidate, expected=expected):
                self.assertTrue(
                    provenance_text_contains_typed_value(
                        candidate, expected, contracts
                    )
                )
        self.assertFalse(
            provenance_text_contains_typed_value(
                "電話番号は090-0000-0046です", 0, [("reissue_count", "integer")]
            )
        )

    def test_spoken_plan_prefix_matches_only_the_plan_name_contract(self) -> None:
        self.assertTrue(
            provenance_text_contains_typed_value(
                "朝が早いので素泊まりに変えてもらえますか",
                "素泊まり山風プラン",
                [("plan_name", "string")],
            )
        )
        self.assertFalse(
            provenance_text_contains_typed_value(
                "朝食付きのままでお願いします",
                "素泊まり山風プラン",
                [("plan_name", "string")],
            )
        )

    def test_spoken_plan_prefix_rejects_a_different_plan(self) -> None:
        for spoken, plan in (("スタンダード20でお願いします", "スタンダード30"), ("おまかせスタンダードにしたい", "おまかせプレミアム"), ("あんしん5年プランで", "あんしん3年プラン")):
            self.assertFalse(provenance_text_contains_typed_value(spoken, plan, [("plan_name", "string")]), spoken)

    def test_declared_spoken_email_ignores_reading_separators(self) -> None:
        self.assertTrue(
            provenance_text_contains_declared_spoken(
                "シー・ユー・エス、アットマーク、イーです。",
                "シーユーエスアットマークイー",
            )
        )

    def test_only_the_called_argument_contract_can_supply_vocabulary(self) -> None:
        scenario = {
            "tools": {
                "register_case_record": {
                    "arguments": [{
                        "name": "handling_type",
                        "description": "取扱種別",
                        "enum": ["change", "escalation"],
                    }]
                }
            }
        }
        source = _recalculate_argument_provenance(
            scenario,
            [],
            tool_id="register_case_record",
            field="handling_type",
            value="escalation",
            call_seq=1,
            contracts=[],
        )
        self.assertEqual("tool_contract", source["derived_source_kind"])
        self.assertEqual(
            "tools.register_case_record.arguments.handling_type.enum",
            source["source_path"],
        )
        unknown = _recalculate_argument_provenance(
            scenario,
            [],
            tool_id="register_case_record",
            field="handling_type",
            value="invented_value",
            call_seq=1,
            contracts=[],
        )
        self.assertEqual("unknown", unknown["derived_source_kind"])

    def test_zero_reissue_count_is_operational_tool_vocabulary(self) -> None:
        source = _recalculate_argument_provenance(
            {
                "tools": {
                    "lookup_receipt": {
                        "arguments": [
                            {
                                "name": "reissue_count",
                                "type": "integer",
                                "required": True,
                            }
                        ]
                    }
                }
            },
            [],
            tool_id="lookup_receipt",
            field="reissue_count",
            value=0,
            call_seq=1,
            contracts=[("reissue_count", "integer")],
        )

        self.assertEqual("tool_contract", source["derived_source_kind"])

    def test_none_argument_is_not_applicable_to_provenance(self) -> None:
        scenario = {
            "scenario_id": "nullable-argument",
            "expected_procedure": {
                "expected_tool_trace": [{"tool": "update", "args": {"note": None}}]
            },
            "tools": {
                "update": {
                    "arguments": [{"name": "note", "type": "string", "required": True}]
                }
            },
        }
        event_log = [
            {
                "seq": 1,
                "actor": "operator",
                "event_type": "tool_call",
                "tool": "update",
                "arguments": {"note": None},
            },
            {
                "seq": 2,
                "actor": "tool",
                "event_type": "tool_result",
                "tool": "update",
                "result": {"ok": True},
            },
        ]

        score = _score_m11(scenario, event_log, _attempts(event_log))
        provenance = score["details"]["sub_facets"]["argument_provenance"]

        self.assertTrue(provenance["passed"])
        self.assertEqual(
            "not_applicable",
            provenance["details"]["fields"][0]["derived_source_kind"],
        )

    @unittest.skipUnless(
        HTL_001_PACKAGE_PATH.is_file() and HTL_001_RECORD_PATH.is_file(),
        "real htl-001 package/record artifacts unavailable",
    )
    def test_real_htl_001_argument_provenance_passes(self) -> None:
        scenario = convert_package(load_package(HTL_001_PACKAGE_PATH), mode="text")
        record = json.loads(HTL_001_RECORD_PATH.read_text(encoding="utf-8"))

        score = score_conversation_log_metrics(
            scenario,
            record["final_world"],
            record["event_log"],
            run_id=record["run_id"],
        )
        provenance = score["M11"]["details"]["sub_facets"]["argument_provenance"]
        fields = provenance["details"]["fields"]

        self.assertTrue(provenance["passed"])
        for field_name in ("planned_checkin_time", "late_arrival_notified"):
            matching_fields = [field for field in fields if field["field"] == field_name]
            self.assertTrue(matching_fields)
            self.assertTrue(
                all(field["derived_source_kind"] != "unknown" for field in matching_fields)
            )

    @unittest.skipUnless(
        HTL_001_PACKAGE_PATH.is_file() and HTL_001_RECORD_PATH.is_file(),
        "real htl-001 package/record artifacts unavailable",
    )
    def test_unmentioned_phone_number_remains_unknown(self) -> None:
        scenario = convert_package(load_package(HTL_001_PACKAGE_PATH), mode="text")
        record = json.loads(HTL_001_RECORD_PATH.read_text(encoding="utf-8"))
        event_log = deepcopy(record["event_log"])
        fabricated_phone = "080-9999-9998"
        verify_call = next(
            event
            for event in event_log
            if event.get("event_type") == "tool_call"
            and event.get("tool") == "verify_identity"
        )
        verify_call["arguments"]["phone_number"] = fabricated_phone

        score = score_conversation_log_metrics(
            scenario,
            record["final_world"],
            event_log,
            run_id=record["run_id"],
        )
        provenance = score["M11"]["details"]["sub_facets"]["argument_provenance"]
        phone_field = next(
            field for field in provenance["details"]["fields"]
            if field["field"] == "phone_number"
        )

        self.assertEqual("unknown", phone_field["derived_source_kind"])
        self.assertFalse(provenance["passed"])


if __name__ == "__main__":
    unittest.main()


def test_kana_columns_match_hiragana_and_katakana() -> None:
    from elyza_agent_tasks_customer_service.evaluation.contracts.type_contracts import _normalized_match_value

    assert _normalized_match_value("そば", "allergen_name_kana", None) == _normalized_match_value("ソバ", "allergen_name_kana", None)
    assert _normalized_match_value("やまだ たろう", "customer_name_kana", None) == "ヤマダタロウ"
