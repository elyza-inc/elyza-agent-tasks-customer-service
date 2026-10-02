"""Permanent checks for package reachability, literal SOP references, and builds."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import (
    PackageRuntime,
)
from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import load_package
from tests import SCENARIO_COUNT
from tests.conftest import ASSEMBLE_SCRIPT, PACKAGE_SOURCES, PACKAGES_DIR


ECF_PHONE_CASES = frozenset(
    {
        "ecf-001", "ecf-002", "ecf-003", "ecf-006", "ecf-007", "ecf-008",
        "ecf-009", "ecf-010", "ecf-011", "ecf-012", "ecf-013", "ecf-014",
        "ecf-015", "ecf-016", "ecf-017", "ecf-018", "ecf-019", "ecf-020",
        "ecf-022",
    }
)
IDENTIFIER_PATTERN = re.compile(r"(?<![a-z])[a-z]+_[a-z_]+(?![a-z])")
FULLWIDTH_ALNUM_PATTERN = re.compile(r"[Ａ-Ｚａ-ｚ０-９]")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
QUESTION_ITEM_KEYS = {
    "氏名": ("name", "caller", "customer", "member", "confirmation"),
    "電話": ("phone", "contact", "callback", "confirmation"),
    "生年月日": ("birthdate", "acquired", "license", "confirmation"),
    "郵便": ("postal", "address", "confirmation"),
    "住所": ("address", "location", "pickup", "postal", "confirmation"),
    "メール": ("email", "mail", "destination", "confirmation"),
    "日付": ("date", "day", "month", "period", "window", "time", "confirmation"),
    "金額": ("amount", "fee", "fare", "price", "claim"),
    "番号": ("number", "id", "reference", "code", "serial", "phone", "contact", "card", "account", "confirmation"),
    "同意": ("consent", "confirmation", "agreement", "ack", "declared", "required"),
}


def _packages(directory: Path) -> dict[str, bytes]:
    """Return package bytes keyed by name; non-package files are ignored."""

    return {path.name: path.read_bytes() for path in sorted(directory.glob("*.yaml"))}


def test_gold_walk_reaches_every_required_mutation() -> None:
    paths = sorted(PACKAGES_DIR.glob("*.yaml"))
    assert len(paths) == SCENARIO_COUNT
    failures = []
    for path in paths:
        package = json.loads(path.read_text(encoding="utf-8"))
        reached = {
            change["entity"]
            for turn in package["gold_dialogue"]["turns"]
            if turn["kind"] == "tool_result"
            for change in turn.get("world_changes", [])
        }
        required = package["contracts"]["completion"]["required_mutations"]
        missing = {item["table"] for item in required} - reached
        if missing:
            failures.append(f"{package['scenario_id']}: gold dialogue misses mutation tables {sorted(missing)}")
    _fail_if_any(failures)


def test_sop_questions_have_persona_values_and_tools_exist() -> None:
    failures = []
    for path in sorted(PACKAGES_DIR.glob("*.yaml")):
        package = json.loads(path.read_text(encoding="utf-8"))
        sops = {
            sop["id"]: sop
            for category in package["sop_catalog"]["categories"].values()
            for sop in category["sops"]
        }
        inputs = [
            item
            for group in ("identity", "declared")
            for item in package["inputs"].get(group, [])
        ]
        input_steps = {item["question_step_id"] for item in inputs}
        for sop_id in package["use_sops"]:
            for step in sops[sop_id]["steps"]:
                if step["kind"] == "tool":
                    if step["tool_id"] not in package["tools"]:
                        failures.append(f"{package['scenario_id']}: absent SOP tool {step['tool_id']}")
        if package["scenario_id"] in ECF_PHONE_CASES:
            if not any(item["name"] == "phone_number" for item in inputs):
                failures.append(f"{package['scenario_id']}: missing phone_number input")
            if "ask_identity" not in input_steps:
                failures.append(f"{package['scenario_id']}: missing ask_identity input step")
    _fail_if_any(failures)


def test_gate_and_regenerated_packages_are_identical(tmp_path: Path) -> None:
    output = tmp_path / "packages"
    for tasks_root, solutions_root in PACKAGE_SOURCES:
        for tasks_dir in sorted(tasks_root.iterdir()):
            if tasks_dir.is_dir():
                subprocess.run(
                    [
                        sys.executable,
                        str(ASSEMBLE_SCRIPT),
                        "--tasks", str(tasks_dir),
                        "--solutions", str(solutions_root / tasks_dir.name),
                        "--output", str(output),
                    ],
                    check=True,
                )
    gate = _packages(PACKAGES_DIR)
    regenerated = _packages(output)
    assert len(gate) == len(regenerated) == SCENARIO_COUNT
    assert hashlib.sha256(b"".join(gate.values())).digest() == hashlib.sha256(
        b"".join(regenerated.values())
    ).digest()


def _loaded_packages() -> list[dict[str, Any]]:
    """Load every assembled object-root package used by these integrity gates."""

    packages = [load_package(path) for path in sorted(PACKAGES_DIR.glob("*.yaml"))]
    assert len(packages) == SCENARIO_COUNT
    return packages


def _fail_if_any(failures: list[str]) -> None:
    """Raise one assertion containing every scenario ID and detailed failure."""

    assert not failures, "\n".join(failures)


def _input_items(package: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for group in ("identity", "declared") for item in package["inputs"][group]]


def _used_sops(package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = {
        sop["id"]: sop
        for category in package["sop_catalog"]["categories"].values()
        for sop in category["sops"]
    }
    return {sop_id: catalog[sop_id] for sop_id in package["use_sops"]}


def _matches_selector(row: dict[str, Any], selector: dict[str, Any]) -> bool:
    return all(row["row_identity"].get(key) == value for key, value in selector.items())


def _mutation_seen(requirement: dict[str, Any], runtime: PackageRuntime) -> bool:
    return any(
        result.get("ok") is not False
        and change.get("operation") == requirement.get("kind", "update")
        and change.get("entity") == requirement["table"]
        and _matches_selector(change, requirement["record"])
        and all(change.get("after", {}).get(key) == value for key, value in requirement["set"].items())
        for call in runtime.tool_log
        for result in [call["result"]]
        for change in result.get("changes", [])
    )


def _matches_filters(package: dict[str, Any], tool: dict[str, Any], arguments: dict[str, Any]) -> list[dict[str, Any]]:
    rows = package["world"][tool["entity"]]
    for rule in tool["filters"]:
        if "scope" in rule:
            continue
        value = arguments.get(rule.get("argument"), rule.get("value"))
        column = rule["column"]
        if rule["operator"] == "eq":
            rows = [row for row in rows if row["values"].get(column) == value]
        elif rule["operator"] == "lte":
            rows = [row for row in rows if row["values"].get(column) is not None and row["values"][column] <= value]
        else:
            rows = [row for row in rows if row["values"].get(column) is not None and row["values"][column] >= value]
    return rows


def _text_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [child for key, item in value.items() if key in {"utterance", "value", "spoken"} for child in _text_values(item)]
    if isinstance(value, list):
        return [child for item in value for child in _text_values(item)]
    return [value] if isinstance(value, str) else []


def test_runtime_gold_walk_preserves_forbidden_rows() -> None:
    failures = []
    for package in _loaded_packages():
        scenario_id = package["scenario_id"]
        runtime = PackageRuntime(deepcopy(package), scenario_id=scenario_id)
        initial_world = deepcopy(runtime.world)
        for turn in package["gold_dialogue"]["turns"]:
            if turn["kind"] == "tool_call":
                runtime.call(turn["tool_id"], deepcopy(turn["arguments"]))
        for requirement in package["contracts"]["completion"]["required_mutations"]:
            if not _mutation_seen(requirement, runtime):
                failures.append(f"{scenario_id}: required mutation not reached: {requirement}")
        for requirement in package["contracts"]["completion"]["forbidden_mutations"]:
            before = [row for row in initial_world[requirement["table"]] if _matches_selector(row, requirement["record"])]
            after = [row for row in runtime.world[requirement["table"]] if _matches_selector(row, requirement["record"])]
            values = requirement.get("set", {})
            if values and any(
                old["values"].get(column) != new["values"].get(column)
                for old, new in zip(before, after)
                for column in values
            ):
                failures.append(f"{scenario_id}: forbidden row changed: {requirement['record']}")
    _fail_if_any(failures)


def test_used_sop_question_wiring_items_and_gold_slots() -> None:
    failures = []
    for package in _loaded_packages():
        scenario_id = package["scenario_id"]
        names_by_step: dict[str, set[str]] = {}
        for item in _input_items(package):
            names_by_step.setdefault(item["question_step_id"], set()).add(item["name"])
        input_names = set().union(*names_by_step.values())
        gold_slots = {
            turn.get("slot_id", "")
            for turn in package["gold_dialogue"]["turns"]
            if turn["kind"] == "utterance" and turn["speaker"] == "user"
        }
        for sop in _used_sops(package).values():
            for step in sop["steps"]:
                if step["kind"] != "question":
                    continue
                step_id = step["step_id"]
                if step_id not in names_by_step:
                    failures.append(f"{scenario_id}: SOP question lacks input wiring: {step_id}")
                if not any(slot.endswith(f"__answer_{step_id}") for slot in gold_slots):
                    failures.append(f"{scenario_id}: SOP question lacks gold user slot: {step_id}")
                for word, key_parts in QUESTION_ITEM_KEYS.items():
                    if word in step["description"] and not any(
                        any(part in name for part in key_parts) for name in input_names
                    ):
                        failures.append(f"{scenario_id}: {step_id} requests {word} without matching input key")
    _fail_if_any(failures)


def test_used_sop_and_m04_tools_are_real_without_self_preconditions() -> None:
    failures = []
    for package in _loaded_packages():
        scenario_id = package["scenario_id"]
        tool_ids = set(package["tools"])
        for sop in _used_sops(package).values():
            for step in sop["steps"]:
                if step["kind"] == "tool" and step["tool_id"] not in tool_ids:
                    failures.append(f"{scenario_id}: used SOP tool is absent: {step['tool_id']}")
        m04 = package["contracts"].get("m04")
        if m04 is not None:
            action = m04["critical_action_tool_id"]
            if action in m04["required_precondition_tool_ids"]:
                failures.append(f"{scenario_id}: M04 action is its own precondition: {action}")
            for tool_id in m04["required_precondition_tool_ids"]:
                if tool_id not in tool_ids:
                    failures.append(f"{scenario_id}: M04 precondition tool is absent: {tool_id}")
    _fail_if_any(failures)


def test_contract_values_and_spoken_values_use_public_normal_forms() -> None:
    failures = []
    for package in _loaded_packages():
        scenario_id = package["scenario_id"]
        for requirement in package["contracts"]["completion"]["required_mutations"]:
            for column, value in requirement["set"].items():
                if isinstance(value, str) and (
                    value.rstrip("。").endswith(("です", "ます"))
                    or FULLWIDTH_ALNUM_PATTERN.search(value)
                    or "できる" in value
                ):
                    failures.append(f"{scenario_id}: non-canonical required set {column}={value!r}")
        for text in _text_values(package["inputs"]) + _text_values(package["persona"]):
            if not EMAIL_PATTERN.fullmatch(text) and IDENTIFIER_PATTERN.search(text):
                failures.append(f"{scenario_id}: internal identifier in spoken value: {text!r}")
    _fail_if_any(failures)


def test_identity_filters_are_unique_and_gold_writes_exclude_forbidden_rows() -> None:
    failures = []
    for package in _loaded_packages():
        scenario_id = package["scenario_id"]
        values = {item["name"]: item["value"] for item in _input_items(package)}
        customer_table = package["world_schema"]["customer_table"]
        for tool in package["tools"].values():
            arguments = {rule["argument"]: values[rule["argument"]] for rule in tool["filters"] if "argument" in rule and rule["argument"] in values}
            complete = all("scope" in rule or "value" in rule or rule["argument"] in arguments for rule in tool["filters"])
            if tool["entity"] == customer_table and arguments and complete:
                matches = _matches_filters(package, tool, arguments)
                if len(matches) != 1:
                    failures.append(f"{scenario_id}: verify filter {tool['id']} resolves {len(matches)} rows")
        forbidden = package["contracts"]["completion"]["forbidden_mutations"]
        for call in package["gold_tool_calls"]:
            tool = package["tools"][call["tool_id"]]
            if tool["operation"] == "filtered_search" or not tool["filters"]:
                continue
            rows = _matches_filters(package, tool, call["arguments"])
            mutation_columns = set((tool.get("mutation") or {}).get("set", {}))
            for requirement in forbidden:
                selector = requirement["record"]
                if requirement["set"] and not mutation_columns & set(requirement["set"]):
                    continue
                if any(_matches_selector(row, selector) for row in rows):
                    failures.append(f"{scenario_id}: gold write {tool['id']} matches forbidden/nearby row {selector}")
    _fail_if_any(failures)
