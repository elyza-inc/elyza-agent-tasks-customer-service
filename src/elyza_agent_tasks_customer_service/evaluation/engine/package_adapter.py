"""Convert package YAML mappings to the current scoring input shape.

The loader accepts one object-root ``.yaml`` or ``.yml`` document. The
converter accepts that decoded mapping and one mode from ``text``,
``audio-text``, or ``audio-audio``. Missing or malformed required fields
raise ``PackageAdapterError`` containing the scenario ID and field path; no
value is inferred or substituted.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import load_metric_inventory
from elyza_agent_tasks_customer_service.evaluation.contracts.variants import normalize_variant
from elyza_agent_tasks_customer_service.evaluation.engine.package_build import _world_hash


VALID_MODES = ("text", "audio-text", "audio-audio")
# Audio metrics left out of a text-mode measurement contract (M25/M26 stay in it as N/A).
TEXT_MODE_EXCLUDED_METRIC_IDS = frozenset(("M20", "M21", "M22", "M23"))
PACKAGE_CONTRACT_METRICS = {"M04": "m04", "M05": "m05", "M20": "m20"}
MEASUREMENT_CONTRACT_SCHEMA_VERSION = "measurement_contract"
MEASUREMENT_CONTRACT_IMPL_VERSION = "measurement_contract_materializer"
CONTRACT_HASH_LENGTH = 24
M04_CONTRACT_VERSION = "m04_customer_trigger_contract"
M05_CONTRACT_VERSION = "m05_obligation_contract"
INTERACTION_OBSERVATION_CONTRACT_VERSION = "interaction_observation_contract"
INTERACTION_QUESTION_CONTRACT_VERSION = "interaction_closed_questions"
JAPANESE_REGISTER_CONTEXT_VERSION = "japanese_register_context"
EXTERNAL_REPRESENTATIVE_INPUT_FIELDS = frozenset(
    ("declared_company_name_kana", "declared_contact_person_name")
)
IN_GROUP_STAFF_INPUT_FIELDS = frozenset(("requested_courier_note",))
FROZEN_REFERENCE_VERSION = "frozen_final_world_reference"
STRICT_SLOT_BY_VALUE_TYPE = {
    "date": "date",
    "money": "amount",
    "phone": "phone_number",
    "phone_number": "phone_number",
}
M04_TRIGGER_TYPES = (
    "customer_pressure",
    "customer_misconception",
    "customer_correction",
    "customer_emotion",
)


class PackageAdapterError(ValueError):
    """A grading input cannot be built without inventing a value."""


def _stop(scenario_id: str, field: str, message: str) -> None:
    raise PackageAdapterError(f"{scenario_id}: {field}: {message}")


def _mapping(value: Any, scenario_id: str, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _stop(scenario_id, field, "mapping is required")
    return value


def _list(value: Any, scenario_id: str, field: str) -> list[Any]:
    if not isinstance(value, list):
        _stop(scenario_id, field, "list is required")
    return value


def _required(mapping: dict[str, Any], key: str, scenario_id: str, field: str = "package") -> Any:
    if key not in mapping:
        _stop(scenario_id, f"{field}.{key}", "value is required")
    return mapping[key]


def _scenario_id(package: Any) -> str:
    if not isinstance(package, dict):
        _stop("unknown", "package", "mapping is required")
    value = package.get("scenario_id")
    if not isinstance(value, str) or not value:
        _stop("unknown", "package.scenario_id", "non-empty string is required")
    return value


def load_package(path: Path) -> dict[str, Any]:
    """Load one object-root YAML package.

    Only ``.yaml`` and ``.yml`` are accepted. JSON, multi-document YAML,
    arrays, scalars, and empty documents raise instead of being coerced.
    """

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise PackageAdapterError(f"unknown: package_path: YAML file is required: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _scenario_id(value)
    return value


def convert_package(
    package: dict[str, Any], *, mode: str, variant_id: object | None = None
) -> dict[str, Any]:
    """Return one package in the shape consumed by the scoring code.

    ``package`` must contain object ``world``, ``final_world``, ``answer``,
    and ``contracts`` fields, list ``use_sops`` and ``gold_tool_calls`` fields,
    and a non-empty ``scenario_id``. Optional ``variant_id`` accepts
    ``baseline`` or ``hard``.
    Invalid modes, variants, and missing nested values raise.
    """

    scenario_id = _scenario_id(package)
    if mode not in VALID_MODES:
        _stop(scenario_id, "mode", f"must be one of {VALID_MODES}")
    world = _mapping(_required(package, "world", scenario_id), scenario_id, "package.world")
    final_world = _mapping(
        _required(package, "final_world", scenario_id), scenario_id, "package.final_world"
    )
    use_sops = _list(
        _required(package, "use_sops", scenario_id), scenario_id, "package.use_sops"
    )
    if not use_sops or any(not isinstance(item, str) or not item for item in use_sops):
        _stop(scenario_id, "package.use_sops", "non-empty string list is required")
    answer = _mapping(_required(package, "answer", scenario_id), scenario_id, "package.answer")
    expected_mutations = _list(
        _required(answer, "expected_mutations", scenario_id, "package.answer"),
        scenario_id,
        "package.answer.expected_mutations",
    )
    contracts = _mapping(
        _required(package, "contracts", scenario_id), scenario_id, "package.contracts"
    )
    for field in PACKAGE_CONTRACT_METRICS.values():
        _required(contracts, field, scenario_id, "package.contracts")
    gold_calls = _list(
        _required(package, "gold_tool_calls", scenario_id),
        scenario_id,
        "package.gold_tool_calls",
    )
    expected_trace = _expected_tool_trace(gold_calls, scenario_id)

    converted = deepcopy(package)
    converted.update(
        {
            "initial_world": deepcopy(world),
            "expected_final_db": {"gold_state": deepcopy(final_world)},
            "expected_sop_path": list(use_sops),
            "expected_tool_path": [row["tool"] for row in expected_trace],
            "expected_procedure": {"expected_tool_trace": expected_trace},
            "expected_mutation_log": deepcopy(expected_mutations),
            "measurement_contract": _measurement_contract(
                package=package,
                contracts=contracts,
                scenario_id=scenario_id,
                mode=mode,
            ),
            "_ideal_conversation": _ideal_conversation(package, scenario_id),
            "m04_customer_trigger_contract": _m04_contract(
                package, contracts["m04"], scenario_id
            ),
            "m05_obligation_contract": _m05_contract(
                package, contracts["m05"], scenario_id
            ),
            "interaction_observation_contract": _interaction_observation_contract(
                contracts, scenario_id
            ),
            "interaction_closed_questions": {
                "schema_version": INTERACTION_QUESTION_CONTRACT_VERSION,
                "instances": [],
            },
            "japanese_register_context": _japanese_register_context(
                package, scenario_id
            ),
        }
    )
    if variant_id is not None:
        converted["variant_id"] = normalize_variant(variant_id)
    return converted


def frozen_reference_from_package(package: dict[str, Any]) -> dict[str, Any]:
    """Return the evaluator-private final-world hash declared by one package.

    ``package`` must be an object-root package with object ``final_world``
    and exact ``world_hashes.algorithm/scope/final`` values. Missing or
    malformed values raise ``PackageAdapterError``; the final world is never
    copied into the converted public scenario.
    """

    scenario_id = _scenario_id(package)
    final_world = _mapping(
        _required(package, "final_world", scenario_id), scenario_id, "package.final_world"
    )
    hashes = _mapping(
        _required(package, "world_hashes", scenario_id), scenario_id, "package.world_hashes"
    )
    if hashes.get("algorithm") != "sha256-canonical-json":
        _stop(scenario_id, "package.world_hashes.algorithm", "unsupported value")
    if hashes.get("scope") != "all_tables_all_rows":
        _stop(scenario_id, "package.world_hashes.scope", "unsupported value")
    digest = hashes.get("final")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _stop(scenario_id, "package.world_hashes.final", "lowercase SHA-256 is required")
    if digest != _world_hash(final_world, scenario_id):
        _stop(
            scenario_id,
            "package.world_hashes.final",
            "does not match row-identity-normalized final_world",
        )
    return {
        "schema_version": FROZEN_REFERENCE_VERSION,
        "scenario_id": scenario_id,
        "final_world_sha256": f"sha256:{digest}",
        "final_world_root_keys": sorted(final_world),
        "source_kind": "runtime_package",
    }


def _ideal_conversation(package: dict[str, Any], scenario_id: str) -> list[dict[str, Any]]:
    dialogue = _mapping(
        _required(package, "gold_dialogue", scenario_id), scenario_id, "package.gold_dialogue"
    )
    turns = _list(
        _required(dialogue, "turns", scenario_id, "package.gold_dialogue"),
        scenario_id,
        "package.gold_dialogue.turns",
    )
    result: list[dict[str, Any]] = []
    for index, value in enumerate(turns):
        turn = _mapping(value, scenario_id, f"package.gold_dialogue.turns[{index}]")
        if turn.get("kind") == "utterance":
            speaker = {"operator": "assistant", "user": "user"}.get(turn.get("speaker"))
            text = turn.get("text")
            if speaker is None or not isinstance(text, str) or not text:
                _stop(scenario_id, f"package.gold_dialogue.turns[{index}]", "invalid utterance")
            result.append({"speaker": speaker, "utterance": text})
        elif turn.get("kind") == "tool_call":
            tool_id = turn.get("tool_id")
            arguments = turn.get("arguments")
            if not isinstance(tool_id, str) or not tool_id or not isinstance(arguments, dict):
                _stop(scenario_id, f"package.gold_dialogue.turns[{index}]", "invalid Tool call")
            result.append(
                {"speaker": "assistant", "tool_call": {"name": tool_id, "arguments": deepcopy(arguments)}}
            )
    if not result:
        _stop(scenario_id, "package.gold_dialogue.turns", "scorable turns are required")
    return result


def _m04_contract(
    package: dict[str, Any], raw: Any, scenario_id: str
) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = _mapping(raw, scenario_id, "package.contracts.m04")
    critical_tool = _required(value, "critical_action_tool_id", scenario_id, "package.contracts.m04")
    prerequisites = _list(
        _required(value, "required_precondition_tool_ids", scenario_id, "package.contracts.m04"),
        scenario_id,
        "package.contracts.m04.required_precondition_tool_ids",
    )
    persona = _mapping(_required(package, "persona", scenario_id), scenario_id, "package.persona")
    trigger_types = [key for key in M04_TRIGGER_TYPES if persona.get(key) is not None]
    if len(trigger_types) != 1:
        _stop(scenario_id, "package.persona", "M04 requires exactly one trigger type")
    return {
        "schema_version": M04_CONTRACT_VERSION,
        "edges": [
            {
                "edge_id": f"{scenario_id}:m04",
                "customer_trigger_type": trigger_types[0],
                "critical_action_tool_id": critical_tool,
                "required_precondition_tool_ids": prerequisites,
            }
        ],
    }


def _m05_contract(
    package: dict[str, Any], raw: Any, scenario_id: str
) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = _mapping(raw, scenario_id, "package.contracts.m05")
    deadline_tool = _required(value, "deadline_tool_id", scenario_id, "package.contracts.m05")
    ideal = _ideal_conversation(package, scenario_id)
    deadline_index = _first_ideal_tool_index(ideal, deadline_tool, scenario_id, "M05 deadline")
    disclosure_index = _m05_disclosure_index(package, scenario_id)
    if deadline_index <= disclosure_index:
        deadline_index = next(
            (
                index
                for index, row in enumerate(ideal[disclosure_index + 1 :], disclosure_index + 1)
                if isinstance(row.get("tool_call"), dict)
            ),
            None,
        )
        if deadline_index is None:
            _stop(scenario_id, "package.gold_dialogue", "M05 disclosure has no later Tool")
    required_event_ids: list[str] = []
    for tool_index, tool_id in enumerate(value.get("required_tool_ids") or []):
        if not isinstance(tool_id, str) or not tool_id:
            _stop(scenario_id, f"package.contracts.m05.required_tool_ids[{tool_index}]", "non-empty string is required")
        required_index = _first_ideal_tool_index(ideal, tool_id, scenario_id, "M05 required")
        if required_index >= deadline_index:
            _stop(
                scenario_id,
                f"package.contracts.m05.required_tool_ids[{tool_index}]",
                "required Tool must precede the deadline in gold dialogue",
            )
        required_event_ids.append(f"ideal_conversation:{required_index}")
    consent_turn = value.get("consent_turn_index")
    protected_tool = value.get("protected_action_tool_id")
    if (consent_turn is None) != (protected_tool is None):
        _stop(scenario_id, "package.contracts.m05", "consent_turn_index and protected_action_tool_id must be declared together")
    consent_event_id = None
    protected_action_event_id = None
    if protected_tool is not None:
        protected_index = _first_ideal_tool_index(ideal, protected_tool, scenario_id, "M05 protected action")
        consent_index = _gold_turns_index_to_ideal_index(package, consent_turn, scenario_id)
        if consent_index >= protected_index:
            _stop(scenario_id, "package.contracts.m05.consent_turn_index", "consent must precede the protected action")
        consent_event_id = f"ideal_conversation:{consent_index}"
        protected_action_event_id = f"ideal_conversation:{protected_index}"
    return {
        "schema_version": M05_CONTRACT_VERSION,
        "obligations": [
            {
                "obligation_id": f"{scenario_id}:m05",
                "deadline_tool_id": deadline_tool,
                "deadline_event_id": f"ideal_conversation:{deadline_index}",
                "required_event_ids": required_event_ids,
                "consent_event_id": consent_event_id,
                "protected_action_event_id": protected_action_event_id,
            }
        ],
    }


def _gold_turns_index_to_ideal_index(
    package: dict[str, Any], turns_index: Any, scenario_id: str
) -> int:
    """gold_dialogue.turns の索引を ideal_conversation の索引へ写像する。

    ideal_conversation は turns から utterance と tool_call だけを抽出した列のため、
    tool_result 等を数えない位置へ読み替える必要がある。"""

    if not isinstance(turns_index, int) or turns_index < 0:
        _stop(scenario_id, "package.contracts.m05.consent_turn_index", "non-negative integer is required")
    dialogue = _mapping(
        _required(package, "gold_dialogue", scenario_id), scenario_id, "package.gold_dialogue"
    )
    turns = _list(
        _required(dialogue, "turns", scenario_id, "package.gold_dialogue"),
        scenario_id,
        "package.gold_dialogue.turns",
    )
    if turns_index >= len(turns):
        _stop(scenario_id, "package.contracts.m05.consent_turn_index", "index exceeds gold dialogue turns")
    ideal_index = -1
    for index, value in enumerate(turns):
        turn = _mapping(value, scenario_id, f"package.gold_dialogue.turns[{index}]")
        if turn.get("kind") in ("utterance", "tool_call"):
            ideal_index += 1
        if index == turns_index:
            if turn.get("kind") not in ("utterance", "tool_call"):
                _stop(
                    scenario_id,
                    "package.contracts.m05.consent_turn_index",
                    "turn is not part of the ideal conversation",
                )
            return ideal_index
    raise AssertionError("unreachable")


def m05_judge_deadline_tool(package: dict[str, Any], scenario_id: str) -> str | None:
    """Return the Tool whose first call bounds the M05 disclosure, or ``None`` for the whole call.

    Mirrors ``_m05_contract``: when gold discloses after the declared deadline Tool,
    the deadline moves to the next gold Tool after the disclosure. A deadline that
    only writes case records is not a change operation, so the disclosure may come
    anywhere in the call.
    """

    m05 = _mapping(package["contracts"]["m05"], scenario_id, "package.contracts.m05")
    deadline_tool = _required(m05, "deadline_tool_id", scenario_id, "package.contracts.m05")
    ideal = _ideal_conversation(package, scenario_id)
    deadline_index = _first_ideal_tool_index(ideal, deadline_tool, scenario_id, "M05 deadline")
    disclosure_index = _m05_disclosure_index(package, scenario_id)
    if deadline_index <= disclosure_index:
        deadline_tool = next(
            (
                row["tool_call"]["name"]
                for row in ideal[disclosure_index + 1 :]
                if isinstance(row.get("tool_call"), dict)
            ),
            None,
        )
    tools = package.get("tools") or {}
    if deadline_tool is None or (tools.get(deadline_tool) or {}).get("entity") == "case_records":
        return None
    return deadline_tool


def _m05_disclosure_index(package: dict[str, Any], scenario_id: str) -> int:
    use_sops = set(package["use_sops"])
    categories = _mapping(package["sop_catalog"], scenario_id, "package.sop_catalog").get("categories")
    if not isinstance(categories, dict):
        _stop(scenario_id, "package.sop_catalog.categories", "mapping is required")
    explain_step_ids = {
        step["step_id"]
        for category in categories.values()
        for sop in category.get("sops", [])
        if isinstance(sop, dict) and sop.get("id") in use_sops
        for step in sop.get("steps", [])
        if isinstance(step, dict) and step.get("kind") == "explain" and isinstance(step.get("step_id"), str)
    }
    dialogue = package["gold_dialogue"]["turns"]
    candidate_texts = [
        turn["text"]
        for turn in dialogue
        if isinstance(turn, dict)
        and turn.get("kind") == "utterance"
        and turn.get("speaker") == "operator"
        and turn.get("slot_id") in {f"{scenario_id}__{step_id}" for step_id in explain_step_ids}
        and isinstance(turn.get("text"), str)
    ]
    ideal = _ideal_conversation(package, scenario_id)
    candidates = [
        index
        for index, row in enumerate(ideal)
        if row.get("speaker") == "assistant" and row.get("utterance") in candidate_texts
    ]
    if len(candidates) != 1:
        _stop(scenario_id, "package.contracts.m05.disclosure", "must bind one explain turn")
    return candidates[0]


def _interaction_observation_contract(
    contracts: dict[str, Any], scenario_id: str
) -> dict[str, Any]:
    m20 = _mapping(contracts["m20"], scenario_id, "package.contracts.m20")
    values = _list(
        _required(m20, "important_values", scenario_id, "package.contracts.m20"),
        scenario_id,
        "package.contracts.m20.important_values",
    )
    slots = []
    for index, item in enumerate(values):
        row = _mapping(item, scenario_id, f"package.contracts.m20.important_values[{index}]")
        slot = STRICT_SLOT_BY_VALUE_TYPE.get(row.get("value_type"))
        if slot is not None and slot not in slots:
            slots.append(slot)
    return {
        "schema_version": INTERACTION_OBSERVATION_CONTRACT_VERSION,
        "strict_no_reask_slots": slots,
    }


def _japanese_register_context(
    package: dict[str, Any], scenario_id: str
) -> dict[str, Any]:
    inputs = _mapping(_required(package, "inputs", scenario_id), scenario_id, "package.inputs")
    world_schema = _mapping(
        _required(package, "world_schema", scenario_id), scenario_id, "package.world_schema"
    )
    identity_policy = _mapping(
        _required(world_schema, "identity_policy", scenario_id, "package.world_schema"),
        scenario_id,
        "package.world_schema.identity_policy",
    )
    identity_columns = _list(
        _required(
            identity_policy,
            "columns",
            scenario_id,
            "package.world_schema.identity_policy",
        ),
        scenario_id,
        "package.world_schema.identity_policy.columns",
    )
    if not identity_columns or not isinstance(identity_columns[0], str) or not identity_columns[0]:
        _stop(
            scenario_id,
            "package.world_schema.identity_policy.columns",
            "customer name column is required first",
        )
    name_column = identity_columns[0]
    input_rows = [
        item
        for group in ("identity", "declared")
        for item in _list(inputs.get(group), scenario_id, f"package.inputs.{group}")
        if isinstance(item, dict)
    ]
    rows = [item for item in input_rows if item.get("name") == name_column]
    if len(rows) != 1 or not isinstance(rows[0].get("value"), str) or not rows[0]["value"]:
        _stop(scenario_id, f"package.inputs.{name_column}", "exactly one customer name is required")
    persons = [
        {
            "person_id": "operator",
            "relation_to_operator": "self",
            "surface_forms": ["オペレーター"],
        },
        {
            "person_id": "customer",
            "relation_to_operator": "customer",
            "surface_forms": ["お客様", rows[0]["value"]],
        },
    ]
    input_names = {item.get("name") for item in input_rows}
    external_declared = bool(input_names & EXTERNAL_REPRESENTATIVE_INPUT_FIELDS)
    staff_declared = bool(input_names & IN_GROUP_STAFF_INPUT_FIELDS)
    if external_declared and not staff_declared:
        persons.append(
            {
                "person_id": "external_representative",
                "relation_to_operator": "out_group",
                "surface_forms": ["担当者"],
            }
        )
    elif staff_declared and not external_declared:
        persons.append(
            {
                "person_id": "staff_representative",
                "relation_to_operator": "in_group",
                "surface_forms": ["担当者"],
            }
        )
    return {
        "schema_version": JAPANESE_REGISTER_CONTEXT_VERSION,
        "communication_direction": "operator_to_customer",
        "speaker_person_id": "operator",
        "addressee_person_id": "customer",
        "persons": persons,
        "required_rule_families": ["closed_misuse", "honorific_direction", "in_out_group"],
    }


def _first_ideal_tool_index(
    ideal: list[dict[str, Any]], tool_id: Any, scenario_id: str, label: str
) -> int:
    if isinstance(tool_id, str) and tool_id:
        for index, row in enumerate(ideal):
            if isinstance(row.get("tool_call"), dict) and row["tool_call"].get("name") == tool_id:
                return index
    _stop(scenario_id, label, "Tool is absent from gold dialogue")
    raise AssertionError("unreachable")


def _expected_tool_trace(gold_calls: list[Any], scenario_id: str) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for index, value in enumerate(gold_calls):
        field = f"package.gold_tool_calls[{index}]"
        call = _mapping(value, scenario_id, field)
        tool_id = _required(call, "tool_id", scenario_id, field)
        arguments = _mapping(
            _required(call, "arguments", scenario_id, field),
            scenario_id,
            f"{field}.arguments",
        )
        if not isinstance(tool_id, str) or not tool_id:
            _stop(scenario_id, f"{field}.tool_id", "non-empty string is required")
        trace.append({"tool": tool_id, "args": deepcopy(arguments)})
    if not trace:
        _stop(scenario_id, "package.gold_tool_calls", "non-empty list is required")
    return trace


def _measurement_contract(
    *,
    package: dict[str, Any],
    contracts: dict[str, Any],
    scenario_id: str,
    mode: str,
) -> dict[str, Any]:
    instances = []
    for source_row in load_metric_inventory()["metric_rows"]:
        metric_id = source_row["metric_id"]
        if mode == "text" and metric_id in TEXT_MODE_EXCLUDED_METRIC_IDS:
            continue
        contract = deepcopy(source_row)
        package_field = PACKAGE_CONTRACT_METRICS.get(metric_id)
        if package_field is not None:
            contract = deepcopy(contracts[package_field])
            if metric_id == "M05" and contract is None:
                contract = {
                    "scenario_applicability": {
                        "applicable": False,
                        "reason": "package_m05_contract_not_declared",
                    }
                }
        instances.append(
            {
                "metric_instance_id": f"{scenario_id}:{metric_id}",
                "metric_id": metric_id,
                "contract": contract,
            }
        )
    digest = hashlib.sha256(
        json.dumps(
            {"mode": mode, "package": package},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": MEASUREMENT_CONTRACT_SCHEMA_VERSION,
        "impl_version": MEASUREMENT_CONTRACT_IMPL_VERSION,
        "contract_id": f"{scenario_id}:{digest[:CONTRACT_HASH_LENGTH]}",
        "metric_instances": instances,
    }
