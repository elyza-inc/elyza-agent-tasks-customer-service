from copy import deepcopy

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import (
    MetricInventoryContractError,
    RETIRED_METRIC_IDS,
    load_metric_inventory,
    validate_metric_inventory,
)


def test_m10_is_retired() -> None:
    inventory = load_metric_inventory()

    assert "M10" in RETIRED_METRIC_IDS
    assert "M10" not in {row["metric_id"] for row in inventory["metric_rows"]}
    assert "M10" in {
        row["metric_id"] for row in inventory["retired_metric_rows"]
    }


def test_unknown_metric_id_is_rejected() -> None:
    inventory = deepcopy(load_metric_inventory())
    inventory["metric_rows"].append({"metric_id": "M99"})

    with pytest.raises(MetricInventoryContractError, match="unknown metric IDs"):
        validate_metric_inventory(inventory)
