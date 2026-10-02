"""Direct boundaries for retained lightweight contracts."""

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts import (
    run_outcome,
    type_contracts,
)


def _judge_member() -> dict:
    return {
        "member_id": "judge-a",
        "family": "family-a",
        "provider": "openai",
        "endpoint": "https://example.invalid",
        "transport": "chat_completions",
        "resolved_model_revision": "gpt-4.1",
        "revision_policy": "fixed",
        "api_key_env": "JUDGE_KEY",
        "sampling": {"temperature": 0.2, "seed": 1, "max_tokens": 32},
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, "boolean"), (1, "integer"), (1.5, "number"), ("x", "string"), (None, "NoneType")],
)
def test_scalar_types_and_provenance_variants(value, expected) -> None:
    assert type_contracts.python_value_type(value) == expected
    assert type_contracts.value_matches_type(value, expected) == (expected != "NoneType")
    assert type_contracts._japanese_number_forms(60) == set()
    assert type_contracts.provenance_text_contains_typed_value(
        "2026年5月2日14時30分、五十人、ゼロキュウゼロ-イチニサンヨン-ゴロクナナハチ",
        "2026-05-02 14:30",
        [("when", "string")],
    )
    assert type_contracts.provenance_text_contains_typed_value("五十人", 50, [("capacity", "integer")])
    assert type_contracts.provenance_text_contains_typed_value(
        "ゼロキュウゼロ-イチニサンヨン-ゴロクナナハチ", "09012345678", [("phone_number", "string")]
    )
    with pytest.raises(ValueError, match="expected integer"):
        type_contracts.normalize_typed_value(True, "integer")


def test_run_outcome_error_and_inference_branches() -> None:
    stored = {"schema_version": "run_outcome", "handling_type": "refusal", "completed": True, "reason": "stored"}
    assert run_outcome.derive_run_outcome(record={"run_outcome": stored}, call_events=[]) == stored
    assert run_outcome.derive_run_outcome(record={}, call_events=[])["reason"] == "run_not_completed"
    assert run_outcome.derive_run_outcome(record={"status": "ok"}, call_events=[])["reason"] == "completion_record_missing"
    assert run_outcome.derive_run_outcome(
        record={"target_reached": True, "final_world": {"nested": [{"record_role": "callback"}]}}, call_events=[]
    )["handling_type"] == "callback_commitment"
    guidance = {"event_type": "tool_result", "result": {"ok": True}, "metadata": {"ticket_operation": {"handling_type": "guidance_inquiry_answer"}}}
    assert run_outcome.derive_run_outcome(record={"status": "ok"}, call_events=[guidance])["handling_type"] == "guidance_inquiry_answer"
    assert not run_outcome._event_succeeded({"result": {"ok": False}})
    with pytest.raises(ValueError, match="completed must be boolean"):
        run_outcome.build_run_outcome_from_facts(completed=1, handling_type=None, reason="x")


def test_scalar_helper_edge_cases(monkeypatch) -> None:
    assert type_contracts.provenance_text_contains_declared_spoken("シー・ユー", "シーユー")
    assert type_contracts._provenance_text_contains_date("5月2日", "2026-05-02")
    assert type_contracts._provenance_text_contains_datetime("2026年5月2日14時", "2026-05-02 14:00")
    assert type_contracts._provenance_text_contains_number("0013です", 13)
    assert not type_contracts._provenance_text_contains_number("x", float("inf"))
    assert type_contracts._provenance_phone("ゼロキュウゼロ、イチ") == "0901"
    assert type_contracts.provenance_text_contains_typed_value(
        "朝が早いので素泊まりに変更", "素泊まり山風プラン", [("plan_name", "string")]
    )
    monkeypatch.setattr(type_contracts.yaml, "safe_load", lambda _text: {})
    with pytest.raises(RuntimeError, match="string list"):
        type_contracts._load_scalar_types()


