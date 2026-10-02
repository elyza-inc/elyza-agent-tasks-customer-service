"""Unit tests for the per-turn customer response format."""

from __future__ import annotations

from elyza_agent_tasks_customer_service.evaluation.engine import package_runtime


def test_customer_response_format() -> None:
    response_format = package_runtime._customer_response_format(
        ["persona_second", "persona_first"]
    )
    properties = response_format["json_schema"]["schema"]["properties"]

    assert properties["persona_fired"]["items"]["enum"] == [
        "persona_first",
        "persona_second",
    ]
    assert properties["message"]["minLength"] == 1

    empty_response_format = package_runtime._customer_response_format([])
    empty_properties = empty_response_format["json_schema"]["schema"]["properties"]
    assert empty_properties["persona_fired"]["items"]["enum"] == []
