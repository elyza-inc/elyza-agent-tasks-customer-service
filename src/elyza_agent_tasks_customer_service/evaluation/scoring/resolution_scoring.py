"""Score M01 by hashing the normalized final-world database."""

from __future__ import annotations

from typing import Any

from elyza_agent_tasks_customer_service.evaluation.contracts.artifact_receipts import canonical_json_bytes, sha256_json
from elyza_agent_tasks_customer_service.evaluation.contracts.frozen_reference import validate_frozen_final_world_reference
from elyza_agent_tasks_customer_service.evaluation.contracts.run_outcome import derive_run_outcome


SCORER_VERSION = "resolution_scoring"
ROW_IDENTITY_FIELDS = ("role", "role_name", "customer", "sequence")


def score_resolution(
    *,
    scenario_id: str,
    record: dict[str, Any],
    call_events: list[dict[str, Any]],
    frozen_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return M01 using one record and one evaluator-private hash reference.

    ``record.final_world`` must be the object-root runtime database.
    ``call_events`` is the frozen call epoch. A completed run requires a
    ``frozen_final_world_reference`` with the exact root-key scope hashed at
    package build; runtime-only root keys are excluded without naming them.
    Incomplete runs fail even when no reference was supplied.
    ``critical_violation_free`` is always ``True``: the runtime records no
    hard fails.
    """

    final_db = record.get("final_world")
    if not isinstance(final_db, dict):
        raise ValueError("record.final_world must be an object")
    run_outcome = derive_run_outcome(record=record, call_events=call_events)
    # An incomplete run fails before its final world is canonicalized.
    if not run_outcome["completed"]:
        return {
            "schema_version": SCORER_VERSION,
            "status": "fail",
            "reason": "run_incomplete",
            "value": {
                "resolution_correct": False,
                "critical_violation_free": True,
            },
            "run_outcome": run_outcome,
            "final_world_sha256": sha256_json(final_db),
            "expected_final_world_sha256": None,
            "final_world_root_keys": None,
            "excluded_runtime_root_keys": [],
        }
    actual_hash = sha256_json(_canonical_final_world(final_db))
    if frozen_reference is None:
        return {
            "schema_version": SCORER_VERSION,
            "status": "N/M",
            "reason": "frozen_final_world_reference_missing",
            "value": None,
            "run_outcome": run_outcome,
            "final_world_sha256": actual_hash,
            "expected_final_world_sha256": None,
            "final_world_root_keys": None,
            "excluded_runtime_root_keys": [],
        }
    validate_frozen_final_world_reference(
        frozen_reference,
        scenario_id=scenario_id,
    )
    hash_scope = frozen_reference["final_world_root_keys"]
    missing = sorted(set(hash_scope) - set(final_db))
    if missing:
        raise ValueError(f"record.final_world is missing frozen root keys: {missing}")
    excluded_root_keys = sorted(set(final_db) - set(hash_scope))
    actual_hash = sha256_json(
        _canonical_final_world({key: final_db[key] for key in hash_scope})
    )
    expected_hash = frozen_reference["final_world_sha256"]
    resolution_correct = actual_hash == expected_hash
    reason = "complete_final_world_hash_mismatch"
    if resolution_correct:
        reason = "complete_final_world_hash_match"
    return {
        "schema_version": SCORER_VERSION,
        "status": "pass" if resolution_correct else "fail",
        "reason": reason,
        "value": {
            "resolution_correct": resolution_correct,
            "critical_violation_free": True,
        },
        "run_outcome": run_outcome,
        "final_world_sha256": actual_hash,
        "expected_final_world_sha256": expected_hash,
        "final_world_root_keys": hash_scope,
        "excluded_runtime_root_keys": excluded_root_keys,
    }


def _canonical_final_world(final_world: dict[str, Any]) -> dict[str, Any]:
    """Sort row-identity tables while preserving opaque ordered arrays.

    A row table is a non-empty array whose rows all contain object-valued
    ``row_identity`` fields. Mixed or duplicate identities raise
    ``ValueError``. Arrays without row identities retain their input order.
    """

    result: dict[str, Any] = {}
    for table, value in final_world.items():
        if not isinstance(value, list) or not value:
            result[table] = value
            continue
        identity_rows = [
            isinstance(row, dict) and isinstance(row.get("row_identity"), dict)
            for row in value
        ]
        if not any(identity_rows):
            result[table] = value
            continue
        if not all(identity_rows):
            raise ValueError(f"record.final_world.{table} mixes row identities")
        identities = [canonical_json_bytes(row["row_identity"]) for row in value]
        if len(identities) != len(set(identities)):
            raise ValueError(f"record.final_world.{table} has duplicate row_identity")
        missing = {
            field
            for row in value
            for field in ROW_IDENTITY_FIELDS
            if row["row_identity"].get(field) is None
        }
        if missing:
            raise ValueError(
                f"record.final_world.{table}.row_identity is missing {sorted(missing)}"
            )
        result[table] = sorted(
            value,
            key=lambda row: tuple(
                str(row["row_identity"][field]) for field in ROW_IDENTITY_FIELDS
            ),
        )
    return result
