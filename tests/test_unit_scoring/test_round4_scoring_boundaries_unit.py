"""Direct offline boundaries left by the scoring coverage sweep."""

from __future__ import annotations

import json

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import ACTIVE_METRIC_IDS
from elyza_agent_tasks_customer_service.evaluation.scoring import package_scoring as package
from elyza_agent_tasks_customer_service.evaluation.scoring import post_call_ticket_scoring as ticket
from elyza_agent_tasks_customer_service.evaluation.scoring import scoring_runner as runner


def _instance(metric_id="M01", contract=None):
    return {"metric_instance_id": f"i-{metric_id}", "metric_id": metric_id, "contract": {} if contract is None else contract}


def _scenario(metric_ids=ACTIVE_METRIC_IDS):
    return {"scenario_id": "s", "measurement_contract": {"schema_version": runner.MEASUREMENT_CONTRACT_SCHEMA_VERSION, "impl_version": runner.MEASUREMENT_CONTRACT_IMPL_VERSION, "contract_id": "c", "metric_instances": [_instance(metric_id, None if metric_id in {"M04", "M05"} else {}) for metric_id in metric_ids]}}


def _record():
    return {"run_id": "r", "initial_world": {}, "final_world": {}, "operator_ticket_artifact": None}


def test_package_input_artifact_and_cli_edges(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="objects"):
        package.score_package_record(package=[], record={}, judge_config={}, cache_dir=tmp_path)
    scenario = {"scenario_id": "s", "measurement_contract": {"contract_id": "c", "metric_instances": [{"metric_id": metric_id, "metric_instance_id": metric_id} for metric_id in package.FAILED_RUNTIME_TEXT_METRIC_IDS]}}
    with pytest.raises(ValueError, match="inventory"):
        package._score_failed_runtime(scenario=scenario, record={"error_message": "x"}, mode="audio")
    assert package._failed_runtime_applicability(metric_id="M19", scenario=scenario, record={"variant_id": "hard"}, mode="text") == (False, "m19_trigger_persona_not_declared")
    monkeypatch.setattr(package, "parse_japanese_turns_or_nm", lambda **_: {"reason": "missing"})
    assert "extended_metric_artifact_diagnostics" in package._with_runtime_artifacts({}, [])
    core_report = {"metric_results": [{"metric_id": metric_id, "metric_instance_id": metric_id} for metric_id in ("M15", "M16", "M19")], "violations": []}
    with pytest.raises(ValueError, match="coverage"):
        package._merge_judge_rows(core_report, {"metric_results": [], "judge": {}})
    monkeypatch.setattr(package, "load_package", lambda _: {})
    monkeypatch.setattr(package, "load_mapping", lambda _: {})
    monkeypatch.setattr(package, "load_judge_config", lambda _: {})
    monkeypatch.setattr(package, "score_package_record", lambda **_: {"ok": 1})
    output = tmp_path / "nested" / "report.json"
    assert package.main(["--package", "p", "--record", "r", "--config", "c", "--cache-dir", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == {"ok": 1}


def test_post_call_payload_enum_error_path(monkeypatch):
    contract = {"schema_version": "post_call_ticket_contract", "comparison_fields": list(ticket.DETERMINISTIC_TICKET_FIELDS), "enum_catalog": {"handling_types": ["x"], "action_codes": ["x"], "id_types": ["x"], "evidence_kinds": ["x"]}}
    artifact = {"status": "submitted", "ticket": {"handling_type": "x", "target_ids": [], "performed_actions": [], "evidence_refs": []}}
    monkeypatch.setattr(ticket, "_validate_deterministic_ticket", lambda *_args, **_kwargs: (_ for _ in ()).throw(ticket.TicketEnumSelectionError("bad enum")))
    row = ticket.score_post_call_ticket(contract=contract, artifact=artifact, initial_world={}, final_world={}, call_events=[])
    assert row["status"] == "fail" and row["reason"] == "bad enum"


def test_runner_remaining_dependency_and_contract_edges(monkeypatch):
    assert runner._extended_metric_rows(scenario={}, record=_record(), instances=[_instance("M01")]) == {}
    monkeypatch.setattr(runner, "score_extended_metrics", lambda **_: (_ for _ in ()).throw(ValueError("parser absent")))
    monkeypatch.setattr(runner, "is_japanese_parser_dependency_error", lambda _exc: True)
    rows = runner._extended_metric_rows(scenario={}, record=_record(), instances=[_instance("M17"), _instance("M16")])
    assert rows["i-M17"]["status"] == "N/M" and rows["i-M16"]["status"] == "contract_invalid"
    assert runner._extended_dependency_nm_row(instance=_instance("M17"))["status"] == "N/M"
    scenario = _scenario(("M01",)); contract = scenario["measurement_contract"]
    for key, value, message in [("schema_version", "bad", "schema"), ("impl_version", "bad", "implementation"), ("contract_id", "", "contract_id"), ("metric_instances", {}, "array")]:
        changed = {**contract, key: value}
        with pytest.raises(ValueError, match=message):
            runner._validate_inputs({"scenario_id": "s", "measurement_contract": changed}, _record(), [])
