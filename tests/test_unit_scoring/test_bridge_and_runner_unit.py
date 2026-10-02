"""Direct branch checks for bridge and runner rows with synthetic reports."""

from __future__ import annotations

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import ACTIVE_METRIC_IDS
from elyza_agent_tasks_customer_service.evaluation.scoring import metric_bridge as bridge
from elyza_agent_tasks_customer_service.evaluation.scoring import scoring_runner as runner


def _instance(metric_id="M01", contract=None):
    return {"metric_instance_id": f"i-{metric_id}", "metric_id": metric_id, "contract": contract or {}}


def _availability_rows(status="pass"):
    return [{"metric_instance_id": f"i-{metric_id}", "metric_id": metric_id, "status": status, "reason": "r"} for metric_id in ACTIVE_METRIC_IDS]


def test_m24_derivation_and_attachment_paths():
    missing = bridge._m24_result_from_m17({"metric_instance_id": "m17", "status": "pass"})
    assert missing["status"] == "contract_invalid"
    passed = bridge._m24_result_from_m17({"metric_instance_id": "m17", "status": "pass", "diagnostics": {"source_scorer": "s"}, "value": {"utterance_assessments": [{"turn_ref": "t", "text": "x", "spoken_style": {"violations": []}}]}})
    assert passed["status"] == "pass"
    failed = bridge._m24_result_from_m17({"metric_instance_id": "m17", "status": "pass", "value": {"utterance_assessments": [{"spoken_style": {"violations": [{"kind": "x"}]}}]}})
    assert failed["status"] == "fail" and failed["violations"]
    with pytest.raises(ValueError, match="missing"):
        bridge._m24_result_from_m17({"metric_instance_id": "m17", "value": {"utterance_assessments": [{}]}})
    assert bridge._m24_result_from_m17({"metric_instance_id": "m17", "value": {"utterance_assessments": []}})["status"] == "N/M"
    with pytest.raises(ValueError, match="violations"):
        bridge._m24_result_from_m17({"metric_instance_id": "m17", "value": {"utterance_assessments": [{"spoken_style": {"violations": {}}}]}})
    report = {"metric_results": [{"metric_id": "M17", "metric_instance_id": "m17", "status": "N/M", "value": {}}, {"metric_id": "M24"}], "violations": [{"metric_id": "M24"}], "not_measured": [{"metric_id": "M24"}], "diagnostics": {}}
    bridge._attach_m24_result(report)
    assert [row["metric_id"] for row in report["metric_results"]] == ["M17", "M24"]
    with pytest.raises(ValueError, match="M17"):
        bridge._attach_m24_result({"metric_results": [], "violations": [], "not_measured": [], "diagnostics": {}})


def test_runner_complete_minimal_contract_exercises_dispatch():
    instances = [_instance(metric_id, None if metric_id in {"M04", "M05"} else {}) for metric_id in ACTIVE_METRIC_IDS]
    scenario = {"scenario_id": "s", "measurement_contract": {"schema_version": runner.MEASUREMENT_CONTRACT_SCHEMA_VERSION, "impl_version": runner.MEASUREMENT_CONTRACT_IMPL_VERSION, "contract_id": "c", "metric_instances": instances}}
    record = {"run_id": "r", "initial_world": {}, "final_world": {}, "operator_ticket_artifact": None}
    report = runner.score_record(scenario=scenario, record=record, event_log=[])
    assert [row["metric_id"] for row in report["metric_results"]] == list(ACTIVE_METRIC_IDS)
    with pytest.raises(ValueError, match="object array"):
        runner._validate_inputs(scenario, record, [None])
