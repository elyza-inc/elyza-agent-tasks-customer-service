"""Direct rows and final-world edge cases for compact scorers."""

from __future__ import annotations

import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring import extended_metric_scoring as extended
from elyza_agent_tasks_customer_service.evaluation.scoring import resolution_scoring as resolution


def _scenario(instances):
    return {"scenario_id": "s", "measurement_contract": {"metric_instances": instances}}


def test_extended_input_validation_and_placeholder_rows():
    with pytest.raises(ValueError, match="scenario must"):
        extended._validate_bridge_inputs(scenario=[], artifacts={})
    with pytest.raises(ValueError, match="unknown"):
        extended._validate_bridge_inputs(scenario=_scenario([]), artifacts={"x": 1})
    instances = [{"metric_instance_id": "m16", "metric_id": "M16", "contract": {}}, {"metric_instance_id": "m19", "metric_id": "M19", "contract": {}}]
    result = extended.score_extended_metrics(scenario=_scenario(instances), artifacts={})
    assert [row["status"] for row in result["metric_results"]] == ["contract_invalid", "contract_invalid"]
    with pytest.raises(ValueError, match="duplicate"):
        extended._validate_bridge_inputs(scenario=_scenario([instances[0], instances[0]]), artifacts={})


def test_extended_m17_missing_invalid_context_and_dependency(monkeypatch):
    instance = {"metric_instance_id": "m17", "metric_id": "M17", "contract": {}}
    assert extended._score_m17(instance=instance, scenario={}, artifacts={})["status"] == "contract_invalid"
    monkeypatch.setattr(extended, "validate_register_context", lambda value: {})
    assert extended._score_m17(instance=instance, scenario={}, artifacts={})["status"] == "N/M"
    monkeypatch.setattr(extended, "score_japanese_register_from_parse", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr(extended, "is_japanese_parser_dependency_error", lambda exc: True)
    assert extended._score_m17(instance=instance, scenario={}, artifacts={"m17_parse_bundle": {}})["reason"].endswith("unavailable")
    assert extended._nm_row(instance, "x")["status"] == "N/M"
    assert extended._contract_invalid_row(instance, "x")["violations"][0]["violation_type"].endswith("invalid")


def test_canonical_final_world_and_resolution_invalid_database():
    assert resolution._canonical_final_world({"x": []}) == {"x": []}
    assert resolution._canonical_final_world({"x": ["opaque"]}) == {"x": ["opaque"]}
    rows = [{"row_identity": {"role": "b", "role_name": "b", "customer": "b", "sequence": 2}}, {"row_identity": {"role": "a", "role_name": "a", "customer": "a", "sequence": 1}}]
    assert resolution._canonical_final_world({"x": rows})["x"][0]["row_identity"]["role"] == "a"
    with pytest.raises(ValueError, match="mixes"):
        resolution._canonical_final_world({"x": [rows[0], {}]})
    with pytest.raises(ValueError, match="duplicate"):
        resolution._canonical_final_world({"x": [rows[0], rows[0]]})
    with pytest.raises(ValueError, match="missing"):
        resolution._canonical_final_world({"x": [{"row_identity": {}}]})
    with pytest.raises(ValueError, match="must be an object"):
        resolution.score_resolution(scenario_id="s", record={"final_world": []}, call_events=[], frozen_reference=None)


def test_resolution_outcomes_and_extended_validation_errors():
    incomplete = resolution.score_resolution(scenario_id="s", record={"final_world": {}, "target_reached": False}, call_events=[], frozen_reference=None)
    assert incomplete["reason"] == "run_incomplete"
    missing = resolution.score_resolution(scenario_id="s", record={"final_world": {"handling_type": "refusal"}, "target_reached": True}, call_events=[], frozen_reference=None)
    assert missing["status"] == "N/M"
    for kwargs, message in [
        ({"scenario": _scenario([{ "metric_instance_id": "m17", "metric_id": "M17", "contract": []}]), "artifacts": {}}, "contract must"),
    ]:
        with pytest.raises(ValueError, match=message):
            extended._validate_bridge_inputs(**kwargs)
