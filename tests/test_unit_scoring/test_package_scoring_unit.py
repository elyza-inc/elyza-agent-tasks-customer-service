"""Direct failed-runtime and merge checks for package scoring."""

from __future__ import annotations

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import ACTIVE_METRIC_IDS
from elyza_agent_tasks_customer_service.evaluation.scoring import package_scoring as package


def _scenario():
    return {"scenario_id": "s", "measurement_contract": {"contract_id": "c", "metric_instances": [{"metric_id": metric_id, "metric_instance_id": f"i-{metric_id}"} for metric_id in ACTIVE_METRIC_IDS]}}


def test_failed_runtime_and_applicability_edges():
    report = package._score_failed_runtime(scenario=_scenario(), record={"error_message": "boom", "variant_id": "baseline"}, mode="audio")
    assert report["not_measured"] and report["diagnostics"]["error_message"] == "boom"
    with pytest.raises(ValueError, match="non-empty"):
        package._score_failed_runtime(scenario=_scenario(), record={"error_message": " "}, mode="audio")
    assert package._failed_runtime_applicability(metric_id="M19", scenario=_scenario(), record={"variant_id": "baseline"}, mode="text") == (False, "m19_baseline_not_applicable")
    hard = package._failed_runtime_applicability(metric_id="M19", scenario=_scenario(), record={"variant_id": "hard"}, mode="text")
    assert hard == (False, "m19_trigger_persona_not_declared")
    with pytest.raises(ValueError, match="baseline.*hard"):
        package._failed_runtime_applicability(metric_id="M19", scenario=_scenario(), record={}, mode="text")


