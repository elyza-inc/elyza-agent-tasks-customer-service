"""Database, mutation, recovery, and provenance checks for ``PackageRuntime``."""

from __future__ import annotations

from copy import deepcopy

import json

import pytest

from elyza_agent_tasks_customer_service.evaluation.engine import package_runtime as runtime_module
from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import PackageRuntime, PackageError


def test_synthetic_world_read_update_create_and_schema_edges(minimal_package: dict) -> None:
    """Exercise empty reads, unmatched updates, update/create mutations, and type repair."""

    runtime = PackageRuntime(minimal_package, scenario_id="engine-unit")
    assert runtime.call("search", {"key": "missing"}) == {"ok": True, "rows": [], "changes": []}
    assert runtime.call("update", {"key": "missing", "status": "new"})["error"] == "no_matching_row"
    assert runtime.call("verify", {"name": "ナマエ"})["ok"] is True
    updated = runtime.call("update", {"key": "one", "status": "new"})
    assert updated["changes"][0]["before"]["status"] == "old"
    assert updated["rows"] == [{"key": "one", "status": "new", "count": 1}]
    created = runtime.call("create", {"key": "two", "status": "fresh"})
    assert created["changes"][0]["operation"] == "create"
    assert runtime.call("search", {"key": 1})["error"] == "schema_repair"


def test_completion_target_requires_one_unique_row(duplicate_minimal_package: dict) -> None:
    """An update-style completion target matching duplicates is refused as a tool error, without writing."""

    runtime = PackageRuntime(duplicate_minimal_package, scenario_id="engine-unit")
    runtime.completion = {"required_mutations": [{"table": "items", "record": {"customer": "c1"}, "set": {"status": "new"}}]}
    before = json.dumps(runtime.world, sort_keys=True)
    assert runtime.call("create", {"key": "ignored", "status": "new"})["error"] == "no_matching_row"
    assert json.dumps(runtime.world, sort_keys=True) == before


def test_real_package_lock_fault_fires_once_then_requires_recovery(htl_package: dict) -> None:
    """The declared edit-lock failure leaves the world unchanged and recovers once."""

    runtime = PackageRuntime(htl_package, scenario_id="htl-001")
    name = next(item["value"] for item in htl_package["inputs"]["identity"] if item["name"] == "guest_name_kana")
    guest_id = next(
        row["values"]["guest_id"]
        for row in htl_package["world"][htl_package["world_schema"]["customer_table"]]
        if row["row_identity"]["role"] == "base" and row["values"]["guest_name_kana"] == name
    )
    args = {"guest_id": guest_id, "stay_date": "2026-09-01", "planned_checkin_time": "20:00", "late_arrival_notified": True}
    failed = runtime.call("update_planned_checkin_time", args)
    assert failed == {"ok": False, "error": htl_package["error_injection"]["error_message"], "retryable": True, "world_unchanged": True}
    assert runtime.tool_log[-1]["world_changed"] is False
    assert runtime.call("release_reservation_edit_lock", {"guest_id": args["guest_id"], "stay_date": args["stay_date"]})["ok"] is True
    retried = runtime.call("update_planned_checkin_time", args)
    assert retried["ok"] is True
    assert runtime.fault.completed is True
    assert sum(row["result"].get("error") == htl_package["error_injection"]["error_message"] for row in runtime.tool_log) == 1


def test_argument_provenance_prefers_recent_tool_result(minimal_package: dict) -> None:
    """Record source kind, sequence, and result path for an argument value."""

    runtime = PackageRuntime(minimal_package, scenario_id="engine-unit")
    events = [
        {"seq": 1, "actor": "user", "event_type": "message", "content": "keyはoneです", "event_ref": "u1"},
        {"seq": 2, "actor": "tool", "event_type": "tool_result", "tool": "search", "result": {"key": "one"}, "event_ref": "t2"},
    ]
    provenance = runtime.argument_provenance("search", {"key": "one"}, events)
    assert provenance["key"] == {"source_kind": "tool_result", "source_seq": 2, "source_ref": "t2", "source_tool": "search", "source_path": "result.key", "required": True}
    assert runtime.argument_provenance("search", {"key": "absent"}, events)["key"]["source_kind"] == "unknown"


def test_runtime_sop_argument_and_schema_repair_boundaries(minimal_package: dict) -> None:
    """Keep exact SOP argument diagnostics and the configured repair limit."""

    runtime = PackageRuntime(minimal_package, scenario_id="engine-unit")
    assert runtime.call(runtime_module.SOP_CATEGORY_SEARCH_TOOL, {"query": "x", "extra": "x"})["violations"] == ["extra: unknown argument"]
    for _ in range(runtime_module.MAX_SCHEMA_REPAIRS):
        assert runtime.call("search", {"unknown": "x"})["error"] == "schema_repair"
    assert runtime.call("search", {"unknown": "x"})["error"] == "invalid_arguments"


def test_runtime_search_limit_mutation_contract_and_provenance_boundaries(minimal_package: dict) -> None:
    """Exercise paging, strict mutation shapes, and malformed provenance entries."""

    package = deepcopy(minimal_package)
    package["world_schema"]["search_policy"]["default_limit"] = 1
    package["world"]["items"].append({"row_identity": {"customer": "c1", "sequence": 2}, "values": {"key": "one", "status": "old", "count": 2}})
    runtime = PackageRuntime(package, scenario_id="engine-unit")
    assert len(runtime.call("search", {"key": "one"})["rows"]) == 1
    for mutation, match in ((None, r"must be exactly \{set"), ({"set": {"status": {"bad": "value"}}}, "exactly from_argument or value")):
        broken = deepcopy(package)
        broken["tools"]["update"]["mutation"] = mutation
        with pytest.raises(PackageError, match=match):
            PackageRuntime(broken, scenario_id="engine-unit")
    runtime.tools["search"]["arguments"] = ["bad"]
    assert runtime.argument_provenance("search", {"key": "one"}, [])["key"]["source_kind"] == "unknown"


def test_runtime_completion_and_sop_provenance_edges(minimal_package: dict) -> None:
    """Preserve create completion behavior and ignore non-tool SOP evidence."""

    package = deepcopy(minimal_package)
    package["contracts"] = {"completion": {"required_mutations": [{"table": "items", "record": {"customer": "new"}, "set": {"key": "new", "status": "new"}, "kind": "create"}]}}
    runtime = PackageRuntime(package, scenario_id="engine-unit")
    assert runtime.call("create", {"key": "new", "status": "new"})["changes"][0]["operation"] == "create"
    event = {"seq": 1, "tool": runtime_module.SOP_DETAIL_TOOL, "event_type": "message", "result": {"sop": {"steps": [{"description": "keyはone"}]}}, "event_ref": "x"}
    assert runtime.argument_provenance("search", {"key": "one"}, [event])["key"]["source_kind"] == "unknown"


def test_create_collision_returns_an_error_without_changing_the_world(minimal_package: dict) -> None:
    """Return the duplicate-create error to the model instead of stopping the run."""

    package = deepcopy(minimal_package)
    package["contracts"] = {"completion": {"required_mutations": [{"table": "items", "record": {"customer": "new"}, "set": {"key": "new", "status": "new"}, "kind": "create"}]}}
    runtime = PackageRuntime(package, scenario_id="engine-unit")
    assert runtime.call("create", {"key": "new", "status": "new"})["ok"] is True
    before = deepcopy(runtime.world)

    result = runtime.call("create", {"key": "new", "status": "new"})

    assert result == {"ok": False, "error": "create_target_already_exists", "message": "create target already exists; existing row was not modified"}
    assert runtime.world == before
    assert runtime.tool_log[-1]["world_changed"] is False


def test_runtime_rejects_boolean_search_maximum(minimal_package: dict) -> None:
    """Reject bool values even though bool is an integer subclass."""

    package = deepcopy(minimal_package)
    package["world_schema"]["search_policy"]["max_limit"] = True
    with pytest.raises(PackageError, match="max_limit"):
        PackageRuntime(package, scenario_id="engine-unit")


def test_runtime_ticket_validation_rejects_incomplete_action_fields() -> None:
    """Reject an action row before accessing any of its required fields."""

    contract = {"enum_catalog": {"handling_types": ["change"], "action_codes": ["act"], "id_types": ["id"], "evidence_kinds": ["event"]}}
    ticket = {"handling_type": "change", "target_ids": [], "performed_actions": [{}], "evidence_refs": [], "refusal_reason": None, "promises": [], "answer_summary": None}
    with pytest.raises(PackageError, match=r"performed_actions\[0\].*fields are invalid"):
        runtime_module._validate_ticket(ticket, "s", contract)


def test_runtime_ticket_generation_checks_finish_repair_and_fallback(monkeypatch) -> None:
    """Keep ticket length handling, all repairs, and fallback provenance observable."""

    monkeypatch.setattr(runtime_module, "apply_chat_provider_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_module, "_ticket_response_format", lambda contract: {"json_schema": {"schema": {}}})
    monkeypatch.setattr(runtime_module, "extract_chat_message", lambda response: {"content": response["content"]})
    monkeypatch.setattr(runtime_module, "_validate_ticket", lambda value, scenario_id, contract: value)

    def generate(responses, partial_record=None):
        iterator = iter(responses)
        monkeypatch.setattr(runtime_module, "chat_with_context_retry", lambda *args, **kwargs: next(iterator))
        return runtime_module._generate_operator_ticket(
            scenario_id="s", endpoint="http://local", model="m", headers={}, timeout_sec=1,
            extra_body={}, conversation=[], call_events=[], ticket_contract={}, transport=lambda *args: {},
            partial_record=partial_record,
        )

    # A ticket the operator cannot produce is scored as no ticket instead of stopping the run.
    length_record = {}
    assert generate([{"choices": [{"finish_reason": "length"}], "content": "ignored"}], length_record)[0] is None
    assert "output token limit reached" in length_record["operator_ticket_failure"]["reason"]
    monkeypatch.setattr(runtime_module, "_parse_response_json", lambda content, label: (_ for _ in ()).throw(ValueError("bad")) if content == "bad" else ({}, True))
    valid = {"choices": [{"finish_reason": "stop"}], "content": "good"}
    assert generate([{"choices": [{"finish_reason": "stop"}], "content": "bad"}] * runtime_module.MAX_TICKET_REPAIRS + [valid])[2] is True
    partial_record = {}
    assert generate([{"choices": [{"finish_reason": "stop"}], "content": "bad"}, valid], partial_record)[0]["status"] == "submitted"
    assert partial_record["ticket_repair_events"][0]["attempt"] == 1
    exhausted_record = {}
    bad = {"choices": [{"finish_reason": "stop"}], "content": "bad"}
    assert generate([bad] * (runtime_module.MAX_TICKET_REPAIRS + 1), exhausted_record)[0] is None
    assert exhausted_record["operator_ticket_failure"] == {"reason": "bad"}


def test_runtime_pure_helpers_cover_json_matching_ranking_and_outcomes() -> None:
    """Keep deterministic ticket/runtime helper behavior independent of a model."""

    parsed, repaired = runtime_module._parse_response_json('{"status":"submitted"}', label="unit")
    assert parsed["status"] == "submitted" and repaired is False
    parsed, repaired = runtime_module._parse_response_json('prefix {"status":"submitted"} suffix', label="unit")
    assert parsed["status"] == "submitted" and repaired is True
    assert runtime_module._matching_value_path({"a": {"b": "x"}}, "x", column="name", value_type=None, contains_text=False, path="row") == "row.a.b"
    assert runtime_module._rank_texts("予約", [("a", "予約の変更"), ("b", "その他")])[0][0] == "a"
    assert runtime_module._compare("A", "A", "eq", column="name", value_type=None) is True
    assert runtime_module._values_include({"a": "b"}, {"a": "b"}) is True
    events = [{"event_type": "tool_result", "result": {"ok": True, "changes": [{"operation": "update", "entity": "items", "row_identity": {"id": "1"}, "after": {"status": "done"}}]}}]
    changes = runtime_module._successful_world_changes(events)
    requirement = {"table": "items", "record": {"id": "1"}, "set": {"status": "done"}}
    assert runtime_module._mutation_observed(requirement, changes)
    assert runtime_module._run_outcome_from_completion({"required_mutations": [requirement], "forbidden_mutations": []}, events)["reason"] == "handling_type_not_observed"
