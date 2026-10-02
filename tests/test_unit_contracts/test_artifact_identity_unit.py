"""Direct checks for persisted-receipt and event-alias contracts."""

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts import (
    artifact_receipts as receipts,
    event_identity,
)


def test_receipt_lookup_rejects_collision_and_missing_request(tmp_path) -> None:
    receipt = receipts.write_content_addressed_bundle(
        root=tmp_path, namespace="judge", request_payload={"id": 1}, raw_response={}, parsed=None, metadata={}
    )
    bundle_dir = tmp_path / receipt["relative_path"]
    (bundle_dir / "raw_response.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable artifact collision"):
        receipts.write_content_addressed_bundle(
            root=tmp_path, namespace="judge", request_payload={"id": 1}, raw_response={}, parsed=None, metadata={}
        )
    (bundle_dir / "request.json").unlink()
    with pytest.raises(ValueError, match="cannot read content-addressed JSON artifact"):
        receipts.find_content_addressed_receipt_by_request(
            root=tmp_path, namespace="judge", request_payload={"id": 1}
        )


def test_event_identity_aligns_declared_interaction_value() -> None:
    scenario = {
        "_ideal_conversation": [{"speaker": "user", "utterance": "090 1234"}, {"tool_call": {"name": "generic"}}],
        "expected_tool_path": ["generic"],
        "interaction_closed_questions": {
            "instances": [{"parameters": {"user_turn_ref": "ideal_conversation:0", "value": "0901234"}}]
        },
    }
    events = event_identity.align_referenced_ideal_event_aliases(
        scenario,
        [{"event_type": "message", "actor": "user", "content": "番号は090-1234"}, {"event_type": "tool_call", "tool": "generic"}],
        {"ideal_conversation:1"},
    )
    assert events[0]["event_aliases"] == ["ideal_conversation:0"]
    assert events[1]["event_aliases"] == ["ideal_conversation:1"]


def test_m05_refs_pin_first_runtime_call_and_last_customer_message() -> None:
    from elyza_agent_tasks_customer_service.evaluation.scoring import core_metric_scoring as core

    obligation = {
        "obligation_id": "o",
        "deadline_tool_id": "update",
        "deadline_event_id": "ideal_conversation:4",
        "required_event_ids": ["ideal_conversation:1"],
        "consent_event_id": "ideal_conversation:3",
        "protected_action_event_id": "ideal_conversation:4",
    }
    contract = {"schema_version": core.M05_CONTRACT_VERSION, "obligations": [obligation]}
    # Gold: lookup, confirm, consent, then update fails, refresh, update, refresh, update (3 calls).
    scenario = {
        "m05_obligation_contract": contract,
        "_ideal_conversation": [
            {"speaker": "user", "utterance": "change it"},
            {"tool_call": {"name": "lookup"}},
            {"speaker": "assistant", "utterance": "confirm?"},
            {"speaker": "user", "utterance": "yes"},
            {"tool_call": {"name": "update"}},
            {"tool_call": {"name": "refresh"}},
            {"tool_call": {"name": "update"}},
            {"tool_call": {"name": "refresh"}},
            {"tool_call": {"name": "update"}},
            {"speaker": "user", "utterance": "thanks"},
        ],
    }
    # Runtime: lookup twice, confirmation merged before lookup, update only twice.
    runtime = [
        {"event_type": "message", "actor": "customer", "content": "change it"},
        {"event_type": "tool_call", "tool": "lookup"},
        {"event_type": "message", "actor": "operator", "content": "confirm?"},
        {"event_type": "message", "actor": "customer", "content": "yes, go ahead"},
        {"event_type": "tool_call", "tool": "lookup"},
        {"event_type": "tool_call", "tool": "update"},
        {"event_type": "tool_call", "tool": "refresh"},
        {"event_type": "tool_call", "tool": "update"},
        {"event_type": "message", "actor": "customer", "content": "thanks"},
    ]
    refs = {"ideal_conversation:1", "ideal_conversation:3", "ideal_conversation:4"}
    events = event_identity.align_referenced_ideal_event_aliases(scenario, runtime, refs)
    assert events[1]["event_aliases"] == ["ideal_conversation:1"]
    assert events[3]["event_aliases"] == ["ideal_conversation:3"]
    assert events[5]["event_aliases"] == ["ideal_conversation:4"]
    assert "event_aliases" not in events[8]
    checks = core.score_m05_obligations(contract=contract, call_events=events)["value"]["obligations"][0]["deterministic_checks"]
    assert checks["deadline_met"] is True
    assert checks["consent_before_protected_action"] is True

    # An unresolved protected action no longer passes the consent check.
    unresolved = core.score_m05_obligations(
        contract=contract, call_events=[{"event_type": "tool_call", "tool": "update"}]
    )["value"]["obligations"][0]["deterministic_checks"]
    assert unresolved["consent_before_protected_action"] is False
    assert unresolved["deadline_met"] is False
