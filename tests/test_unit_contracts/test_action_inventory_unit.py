"""Direct boundaries for action execution and metric inventory contracts."""

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts import (
    metric_inventory,
)


def test_metric_inventory_and_forbidden_overall_score() -> None:
    inventory = metric_inventory.load_metric_inventory()
    assert metric_inventory.validate_metric_inventory(inventory)["inventory_id"]
    metric_inventory.assert_no_single_overall_score({"rows": [{"metric_id": "M01"}]})
    with pytest.raises(metric_inventory.MetricInventoryContractError, match="single overall score"):
        metric_inventory.assert_no_single_overall_score({"overall_score": 1})
