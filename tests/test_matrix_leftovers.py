"""Final atomic field cases left unmarked by the matrix implementation records."""

from __future__ import annotations

from copy import deepcopy

import pytest

from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import (
    convert_package,
)
from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import (
    PackageRuntime,
    PackageError,
)
from tests.helpers.gold_record_builder import load_source_package


SCENARIO = ("hotel", "htl-001")
CHANGED = "matrix-changed"


def _package() -> dict:
    """Load the representative object-root package used by final matrix checks."""

    return load_source_package(*SCENARIO)


@pytest.mark.parametrize("group", ("identity", "declared"))
@pytest.mark.parametrize("field", ("name", "question", "question_step_id"))
def test_t03_t04_input_fields_reach_customer_context(group: str, field: str) -> None:
    """台帳A/T03,T04: each previously unmarked input leaf reaches customer context."""

    package = _package()
    package["inputs"][group][0][field] = CHANGED

    assert PackageRuntime(package, scenario_id=package["scenario_id"]).customer_context()["inputs"][group][0][field] == CHANGED


def test_t06_domain_name_reaches_operator_prompt() -> None:
    """台帳A/T06: domain.name is embedded in the operator prompt."""

    package = _package()
    package["domain"]["name"] = CHANGED

    assert CHANGED in PackageRuntime(package, scenario_id=package["scenario_id"]).operator_system_prompt()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("default_limit", 0, "positive integer is required"),
        ("max_limit", 0, "must be at least default_limit"),
        ("order_by", "values", "must be row_identity"),
    ),
)
def test_t12_search_policy_fields_are_runtime_contracts(
    field: str, value: int | str, message: str
) -> None:
    """台帳A/T12: every read search-policy leaf has a stable runtime constraint."""

    package = _package()
    package["world_schema"]["search_policy"][field] = value

    with pytest.raises(PackageError, match=message):
        PackageRuntime(package, scenario_id=package["scenario_id"])


def test_t16_required_and_description_reach_tool_schema() -> None:
    """台帳A/T16: remaining argument leaves alter the emitted tool contract."""

    package = _package()
    argument = package["tools"]["verify_identity"]["arguments"][0]
    argument.update(required=False, description=CHANGED)
    definition = next(
        item for item in PackageRuntime(package, scenario_id=package["scenario_id"]).tool_definitions()
        if item["function"]["name"] == "verify_identity"
    )["function"]

    assert argument["name"] not in definition["parameters"]["required"]
    assert definition["parameters"]["properties"][argument["name"]]["description"] == CHANGED


def test_t25_task_metadata_canary_is_runtime_unread() -> None:
    """台帳A/T25: task metadata.canary does not affect the runtime projection."""

    baseline = _package()
    candidate = deepcopy(baseline)
    candidate["metadata"] = {"canary": CHANGED}

    assert PackageRuntime(candidate, scenario_id=candidate["scenario_id"]).customer_context() == PackageRuntime(
        baseline, scenario_id=baseline["scenario_id"]
    ).customer_context()


@pytest.mark.parametrize("field", ("customer", "role", "role_name", "sequence"))
def test_s13_final_world_identity_fields_reach_adapter(field: str) -> None:
    """台帳A/S13: each remaining final-world identity leaf reaches expected_final_db."""

    package = _package()
    candidate = deepcopy(package)
    table = next(iter(candidate["final_world"]))
    candidate["final_world"][table][0]["row_identity"][field] = CHANGED

    assert convert_package(candidate, mode="text")["expected_final_db"] != convert_package(
        package, mode="text"
    )["expected_final_db"]
