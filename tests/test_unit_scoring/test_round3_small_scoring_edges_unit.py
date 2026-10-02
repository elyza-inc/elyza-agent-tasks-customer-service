"""Remaining small scorer validation and failure-path checks."""

from __future__ import annotations

import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring import core_metric_scoring as core
from elyza_agent_tasks_customer_service.evaluation.scoring import extended_metric_scoring as extended
from elyza_agent_tasks_customer_service.evaluation.scoring import post_call_ticket_scoring as ticket
from elyza_agent_tasks_customer_service.evaluation.scoring import scoring_runner as runner


def _instance(metric_id="M01", contract=None):
    return {"metric_instance_id": f"i-{metric_id}", "metric_id": metric_id, "contract": {} if contract is None else contract}


def _record(**changes):
    result = {"run_id": "r", "initial_world": {}, "final_world": {}, "operator_ticket_artifact": None}
    result.update(changes)
    return result


def _scenario(instances):
    return {"scenario_id": "s", "measurement_contract": {"schema_version": runner.MEASUREMENT_CONTRACT_SCHEMA_VERSION, "impl_version": runner.MEASUREMENT_CONTRACT_IMPL_VERSION, "contract_id": "c", "metric_instances": instances}}


@pytest.mark.parametrize(
    ("scenario", "record", "events", "match"),
    [
        (None, _record(), [], "scenario must"), (_scenario([]), None, [], "record must"),
        (_scenario([]), _record(), [None], "event_log"),
        ({"scenario_id": ""}, _record(), [], "scenario_id"),
        (_scenario([]), {}, [], "missing"),
        (_scenario([]), _record(run_id=""), [], "record.run_id"),
        (_scenario([]), _record(initial_world=[]), [], "initial_world"),
        (_scenario([]), _record(final_world=[]), [], "final_world"),
        (_scenario([]), _record(operator_ticket_artifact=[]), [], "operator_ticket"),
    ],
)
def test_runner_input_root_validation(scenario, record, events, match):
    with pytest.raises(ValueError, match=match):
        runner._validate_inputs(scenario, record, events)


@pytest.mark.parametrize(
    ("instance", "match"),
    [({"metric_instance_id": "", "metric_id": "M01", "contract": {}}, "instance_id"),
     ({"metric_instance_id": "x", "metric_id": "bad", "contract": {}}, "metric_id"),
     ({"metric_instance_id": "x", "metric_id": "M01", "contract": []}, "contract")],
)
def test_runner_measurement_instance_validation(instance, match):
    scenario = _scenario(instance if isinstance(instance, list) else [instance])
    with pytest.raises(ValueError, match=match):
        runner._validate_inputs(scenario, _record(), [])


def test_runner_extended_ticket_and_log_error_rows(monkeypatch):
    instance = _instance("M17")
    monkeypatch.setattr(runner, "score_extended_metrics", lambda **kwargs: (_ for _ in ()).throw(ValueError("bad artifact")))
    assert runner._extended_metric_rows(scenario=_scenario([instance]), record=_record(), instances=[instance])["i-M17"]["status"] == "contract_invalid"
    monkeypatch.setattr(runner, "score_post_call_ticket", lambda **kwargs: (_ for _ in ()).throw(ValueError("bad ticket")))
    assert runner._ticket_metric_row(instance=_instance("M14"), scenario=_scenario([_instance("M14")]), record=_record(), call_events=[], context={"ticket_score": None})["status"] == "N/M"
    assert runner._log_metric_row(instance=_instance("M09"), scenario={}, record=_record(), call_events=[], context={"m09_score": None, "m11_score": None})["status"] == "N/M"


def test_extended_validation_boundaries_and_successful_m17(monkeypatch):
    for kwargs, match in [
        ({"scenario": {}, "artifacts": []}, "artifacts"),
        ({"scenario": {}, "artifacts": {}}, "measurement_contract"),
    ]:
        with pytest.raises(ValueError, match=match):
            extended._validate_bridge_inputs(**kwargs)
    instance = _instance("M17")
    monkeypatch.setattr(extended, "validate_register_context", lambda value: {"x": 1})
    monkeypatch.setattr(extended, "load_rule_catalog", lambda: {})
    monkeypatch.setattr(extended, "score_japanese_register_from_parse", lambda **kwargs: {"status": "pass", "reason": "ok", "schema_version": "s", "receipt": {"parse_receipt_id": "r"}, "rule_families": [{"violation_count": 1}], "utterance_assessments": [{"violation_count": 1}]})
    row = extended._score_m17(instance=instance, scenario={}, artifacts={"m17_parse_bundle": {}})
    assert len(row["violations"]) == 2


def test_ticket_and_core_failure_edges(monkeypatch):
    contract = {"schema_version": "post_call_ticket_contract", "comparison_fields": list(ticket.DETERMINISTIC_TICKET_FIELDS), "enum_catalog": {"handling_types": ["x"], "action_codes": ["x"], "id_types": ["x"], "evidence_kinds": ["x"]}}
    invalid = {"handling_type": "x", "target_ids": [], "performed_actions": [], "evidence_refs": []}
    monkeypatch.setattr(ticket, "build_ticket_fact", lambda *args, **kwargs: invalid)
    assert ticket.score_post_call_ticket(contract=contract, artifact={"status": "submitted", "ticket": invalid}, initial_world={}, final_world={}, call_events=[])["status"] == "pass"
    bad = dict(invalid, handling_type="bad")
    assert ticket.score_post_call_ticket(contract=contract, artifact={"status": "submitted", "ticket": bad}, initial_world={}, final_world={}, call_events=[])["status"] == "fail"
    assert core._m05_deterministic_checks(obligation={"deadline_event_id": "d", "required_event_ids": ["r"], "consent_event_id": "c", "protected_action_event_id": "a"}, event_positions={"d": 2, "r": 3, "c": 1, "a": 6}, trigger_position=0)["deadline_met"] is False


