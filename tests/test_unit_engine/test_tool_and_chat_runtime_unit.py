"""Direct unit tests for tool-schema compilation and chat helper boundaries."""

from __future__ import annotations

from copy import deepcopy

import pytest

from elyza_agent_tasks_customer_service.evaluation.engine import package_adapter as adapter


def _scenario() -> dict:
    return {"scenario_id": "engine-tools", "tool_registry": "unit", "initial_world": {"records": [{"record_id": "r1", "customer_id": "c1"}]}, "tools": [{"name": "lookup", "description": "照会", "operation_kind": "filtered_search", "primary_key": "record_id", "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}}]}


def _execution_package() -> dict:
    return {
        "scenario_id": "engine-tools",
        "initial_db": {"records": [{"record_id": "r1", "customer_id": "c1"}]},
        "tools": [
            {
                "source_tool_id": "lookup",
                "effect": {
                    "operation_kind": "filtered_search",
                    "primary_key": "record_id",
                    "entity_id": "records",
                    "execution_spec": {
                        "mutation": {"kind": "none"},
                        "v6_filters": [{"operator": "eq", "input_name": "customer_id", "field": "customer_id"}],
                        "outputs": [{"source": "selected_records", "output_name": "records"}],
                    },
                },
            }
        ],
    }


def _register_package(*, caller_role: str | None = None, duplicate_name: bool = False) -> dict:
    rows = [{"name": "name", "value": "山田", "question": "お名前", "question_step_id": "q"}]
    if duplicate_name:
        rows.append({"name": "name", "value": "別名", "question": "お名前", "question_step_id": "q2"})
    if caller_role is not None:
        rows.append({"name": "caller_role", "value": caller_role, "question": "立場", "question_step_id": "role"})
    return {"inputs": {"identity": rows, "declared": []}, "world_schema": {"identity_policy": {"columns": ["name"]}}}


def test_package_adapter_rejects_non_string_sop_ids(htl_package: dict) -> None:
    """Keep the package SOP list a non-empty string-only contract."""

    package = deepcopy(htl_package)
    package["use_sops"] = [1]
    with pytest.raises(adapter.PackageAdapterError, match="package.use_sops"):
        adapter.convert_package(package, mode="text")
