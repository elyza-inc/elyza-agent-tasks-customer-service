"""台帳Aの未実装 ``data/tasks`` field cases を現行ランタイムで固定する。"""

from __future__ import annotations

import re
from typing import Any

import pytest

from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import (
    PackageRuntime,
    PackageError,
)
from tests.helpers.gold_record_builder import load_source_package


HOTEL_CASE = ("hotel", "htl-001")
REFERENCE_DATE_CASE = ("hotel", "htl-010")
PERSONA_FIELDS = (
    "concern",
    "corrected_value",
    "critical_action_tool_id",
    "deadline",
    "deadline_tool_id",
    "declared_input_name",
    "escalation_requested",
    "escalation_utterance",
    "explicit_acknowledgement_required",
    "fires_when",
    "incorrect_value",
    "required_precondition_tool_ids",
    "statement",
    "stop_condition",
    "timing",
    "trigger",
    "utterance",
)


def _package(domain: str, scenario_id: str) -> dict[str, Any]:
    """Load one object-root task/solution package for one exact scenario ID."""

    return load_source_package(domain, scenario_id)


def _runtime(package: dict[str, Any]) -> PackageRuntime:
    """Build the task runtime using the package's declared scenario ID."""

    return PackageRuntime(package, scenario_id=package["scenario_id"])


def _remove(value: dict[str, Any], *path: str | int) -> None:
    """Delete one fixture field using an exact object/list path."""

    current: Any = value
    for part in path[:-1]:
        current = current[part]
    del current[path[-1]]


def _runtime_error(
    package: dict[str, Any], field: str, reason: str
) -> pytest.RaisesExc[PackageError]:
    """Require the runtime's exact scenario ID, field path, and reason text."""

    message = f"{package['scenario_id']}: {field}: {reason}"
    return pytest.raises(PackageError, match=f"^{re.escape(message)}$")


def _first_with_key(items: list[Any], key: str) -> tuple[int, dict[str, Any]]:
    """Return the first object item carrying one requested fixture key."""

    for index, item in enumerate(items):
        if isinstance(item, dict) and key in item:
            return index, item
    raise AssertionError(f"no item contains {key}")


@pytest.mark.parametrize("field", ("date",))
def test_t05_reference_date_fields_reach_customer_context(field: str) -> None:
    """| T05 | `inputs.reference_date_spec.{date,departure_date,departure_entity_id,departure_record_id,derived_values.days_until_departure}` | 5 | R1 (`:601-610`), R2 (`:839`) | customer prompt、日付tool引数、M11 |  |"""

    package = _package(*REFERENCE_DATE_CASE)
    _remove(package, "inputs", "reference_date_spec", field)

    context = _runtime(package).customer_context()
    assert field not in context["inputs"]["reference_date_spec"], package["scenario_id"]


@pytest.mark.parametrize(
    ("ledger_id", "path", "marker"),
    (
        ("T07", ("domain", "policies", "no_show_rule", "description"), "無連絡の場合"),
        ("T08", ("domain", "escalation_routes", "front_manager", "name"), "フロント責任者"),
        ("T08", ("domain", "escalation_routes", "front_manager", "conditions"), "本人照合が完全一致しない"),
    ),
)
def test_t07_t08_domain_fields_reach_operator_prompt(
    ledger_id: str, path: tuple[str, ...], marker: str
) -> None:
    """| T07 | `domain.policies.*.*` | 1 | R2 (`:832`, object全体) | operator規定、会話系指標 |  |\n\n| T08 | `domain.escalation_routes.*.{name,conditions[]}` | 2 | R2 (`:832`, object全体) | M11/M14/M15/M19 |  |"""

    package = _package(*HOTEL_CASE)
    _remove(package, *path)

    prompt = _runtime(package).operator_system_prompt()
    assert marker not in prompt, f"{ledger_id}:{package['scenario_id']}"
    assert "ドメイン共通規定" in prompt, package["scenario_id"]


@pytest.mark.parametrize("field", ("customer", "role", "role_name", "sequence"))
def test_t13_row_identity_fields_reach_mutation_change(field: str) -> None:
    """| T13 | `world.*[].row_identity.{customer,role,role_name,sequence}` | 4 | R1 (`:637-647`), R2 (`:1269-1424`) | M01、M14、tool結果 |  |"""

    package = _package(*HOTEL_CASE)
    package.pop("error_injection")
    row = package["world"]["reservations"][0]
    _remove(row, "row_identity", field)
    runtime = _runtime(package)
    call = package["gold_tool_calls"][2]
    result = runtime.call(call["tool_id"], call["arguments"])

    assert result["ok"] is True, call["tool_id"]
    assert field not in result["changes"][0]["row_identity"], package["scenario_id"]


def test_t14_world_values_field_changes_target_search_result() -> None:
    """| T14 | `world.*[].values.*` | 1 | R1/R2 (`:642-647,1269-1424`) | M01、M11、M14 |  |"""

    package = _package(*HOTEL_CASE)
    _remove(package["world"]["hotel_guest"][0], "values", "guest_name_kana")
    runtime = _runtime(package)
    call = package["gold_tool_calls"][0]
    result = runtime.call(call["tool_id"], call["arguments"])

    assert result == {"ok": True, "rows": [], "changes": []}, call["tool_id"]


@pytest.mark.parametrize(
    ("field", "reason"),
    (
        ("id", "value is required"),
        ("operation", "value is required"),
        ("entity", "value is required"),
        ("description", "non-empty string is required"),
        ("returns", "non-empty string is required"),
    ),
)
def test_t15_tool_fields_have_exact_runtime_errors(field: str, reason: str) -> None:
    """| T15 | `tools.*.{id,operation,entity,description,returns}` | 5 | R1 (`:648-710`), R2 (`:879-919,1269-1351`) | M04/M05/M11/M14 |  |"""

    package = _package(*HOTEL_CASE)
    tool_id = "verify_identity"
    _remove(package["tools"][tool_id], field)

    error_field = f"tools.{tool_id}.{field}"
    with _runtime_error(package, error_field, reason):
        runtime = _runtime(package)
        if field in {"description", "returns"}:
            runtime.tool_definitions()


@pytest.mark.parametrize("field", ("argument", "column", "operator", "value"))
def test_t17_filter_fields_have_exact_runtime_errors(field: str) -> None:
    """| T17 | `tools.*.filters[].{argument,column,operator,value}` | 4 | R1 (`:675-702`), R2 (`:1269-1317`) | tool検索/更新、M01/M11 |  |"""

    package = _package(*HOTEL_CASE)
    selected_tool = None
    selected_index = None
    for tool_id, tool in package["tools"].items():
        index, rule = _first_with_key(tool["filters"], field) if any(
            isinstance(item, dict) and field in item for item in tool["filters"]
        ) else (None, None)
        if rule is not None:
            selected_tool, selected_index = tool_id, index
            break
    assert selected_tool is not None and selected_index is not None, field
    _remove(package["tools"][selected_tool]["filters"][selected_index], field)

    target = f"tools.{selected_tool}.filters[{selected_index}]"
    if field == "column":
        with _runtime_error(package, f"{target}.column", "value is required"):
            _runtime(package)
    else:
        with _runtime_error(
            package,
            target,
            "must contain exactly column, operator, and argument or value",
        ):
            _runtime(package)


@pytest.mark.parametrize("field", ("from_argument", "value"))
def test_t18_mutation_source_fields_have_exact_runtime_errors(field: str) -> None:
    """| T18 | `tools.*.mutation.set.*.{from_argument,value}` | 2 | R1 (`:703-812`), R2 (`:1325-1351`) | M01/M11/M14 |  |"""

    package = _package(*HOTEL_CASE)
    for tool_id, tool in package["tools"].items():
        for column, source in tool.get("mutation", {}).get("set", {}).items():
            if field in source:
                _remove(source, field)
                with _runtime_error(
                    package,
                    f"tools.{tool_id}.mutation.set.{column}",
                    "must contain exactly from_argument or value",
                ):
                    _runtime(package)
                return
    raise AssertionError(f"no mutation source contains {field}")


def test_t19_category_title_has_exact_runtime_error() -> None:
    """| T19 | `sop_catalog.categories.*.title` | 1 | R1 (`:743-756`), R2 (`:1017-1051`) | M09、operator SOP選択 |  |"""

    package = _package(*HOTEL_CASE)
    category_id = "reservation_schedule"
    _remove(package["sop_catalog"]["categories"][category_id], "title")

    with _runtime_error(
        package,
        f"sop_catalog.categories.{category_id}.title",
        "value is required",
    ):
        _runtime(package)


@pytest.mark.parametrize(
    ("field", "reason"),
    (
        ("id", "value is required"),
        ("title", "value is required"),
        ("applicability", "value is required"),
        ("completion", "value is required"),
        ("prohibition", "value is required"),
        ("escalation", "value is required"),
        ("branching", None),
    ),
)
def test_t20_sop_fields_reach_validation_or_sop_result(
    field: str, reason: str | None
) -> None:
    """| T20 | `sop_catalog.categories.*.sops[].{id,title,applicability,branching,completion,prohibition,escalation}` | 7 | R1 (`:757-769`), R2 (`:1017-1108`), R3 (`:460-490`) | M05/M09/M11 |  |"""

    package = _package(*HOTEL_CASE)
    category = package["sop_catalog"]["categories"]["reservation_schedule"]
    index, sop = _first_with_key(category["sops"], field)
    sop_id = sop["id"]
    _remove(sop, field)

    target = f"sop_catalog.categories.reservation_schedule.sops[{index}]"
    if reason is not None:
        with _runtime_error(package, f"{target}.{field}", reason):
            _runtime(package)
    else:
        runtime = _runtime(package)
        runtime.available_sop_ids.add(sop_id)
        result = runtime.call("get_sop", {"sop_id": sop_id})
        assert result["ok"] is True, sop_id
        assert field not in result["sop"], package["scenario_id"]


@pytest.mark.parametrize(
    ("field", "kind", "reason"),
    (
        ("kind", "question", "value is required"),
        ("step_id", "question", "value is required"),
        ("description", "question", "value is required"),
        ("tool_id", "tool", "value is required"),
    ),
)
def test_t21_sop_step_fields_have_exact_runtime_errors(
    field: str, kind: str, reason: str
) -> None:
    """| T21 | `sop_catalog.categories.*.sops[].steps[].{kind,step_id,description,tool_id}` | 4 | R1 (`:770-780`), R2 (`:1017-1108`), R3 (`:465-480`) | M05/M09/M11 |  |"""

    package = _package(*HOTEL_CASE)
    steps = package["sop_catalog"]["categories"]["reservation_schedule"]["sops"][0]["steps"]
    index = next(index for index, step in enumerate(steps) if step["kind"] == kind)
    _remove(steps[index], field)
    target = f"sop_catalog.categories.reservation_schedule.sops[0].steps[{index}].{field}"

    with _runtime_error(package, target, reason):
        _runtime(package)


@pytest.mark.parametrize(
    ("persona_name", "domain", "scenario_id"),
    (
        ("customer_pressure", "hotel", "htl-001"),
        ("customer_misconception", "hotel", "htl-011"),
        ("customer_correction", "hotel", "htl-017"),
        ("customer_emotion", "hotel", "htl-007"),
    ),
)
def test_t22_persona_null_branch_reaches_customer_context(
    persona_name: str, domain: str, scenario_id: str
) -> None:
    """| T22 | `persona.{customer_pressure,customer_misconception,customer_correction,customer_emotion}` のnull/非null4分岐、および非null objectの `{concern,corrected_value,critical_action_tool_id,deadline,deadline_tool_id,declared_input_name,escalation_requested,escalation_utterance,explicit_acknowledgement_required,fires_when,incorrect_value,required_precondition_tool_ids,statement,stop_condition,timing,trigger,utterance}` | 21 | root/`utterance`/`fires_when`: R1 (`:781-786`), R3 (`:342-345,740-807`); object全体: R2 (`:840,1604-1695`) | M04/M05/M19、customer simulator |  |"""

    package = _package(domain, scenario_id)
    package["persona"][persona_name] = None

    persona = _runtime(package).customer_context()["persona"]
    assert persona[persona_name] is None, f"{scenario_id}:{persona_name}"


@pytest.mark.parametrize("field", PERSONA_FIELDS)
def test_t22_persona_object_fields_reach_customer_context(field: str) -> None:
    """| T22 | `persona.{customer_pressure,customer_misconception,customer_correction,customer_emotion}` のnull/非null4分岐、および非null objectの `{concern,corrected_value,critical_action_tool_id,deadline,deadline_tool_id,declared_input_name,escalation_requested,escalation_utterance,explicit_acknowledgement_required,fires_when,incorrect_value,required_precondition_tool_ids,statement,stop_condition,timing,trigger,utterance}` | 21 | root/`utterance`/`fires_when`: R1 (`:781-786`), R3 (`:342-345,740-807`); object全体: R2 (`:840,1604-1695`) | M04/M05/M19、customer simulator |  |"""

    for domain, scenario_id in (("hotel", f"htl-{number:03d}") for number in range(1, 25)):
        package = _package(domain, scenario_id)
        for persona_name, value in package["persona"].items():
            if isinstance(value, dict) and field in value:
                _remove(value, field)
                if field in {"utterance", "fires_when"}:
                    with _runtime_error(
                        package,
                        f"persona.{persona_name}.{field}",
                        "value is required",
                    ):
                        _runtime(package)
                else:
                    persona = _runtime(package).customer_context()["persona"]
                    assert field not in persona[persona_name], f"{scenario_id}:{persona_name}"
                return
    raise AssertionError(f"no persona object contains {field}")
