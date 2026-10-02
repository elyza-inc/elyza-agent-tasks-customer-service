"""Validate the evaluator-private frozen final-world hash reference of one package.

Values are validated but never copied into a public scenario or model prompt.
"""

from __future__ import annotations

from typing import Any


FROZEN_REFERENCE_VERSION = "frozen_final_world_reference"
SHA256_PREFIX = "sha256:"


def validate_frozen_final_world_reference(
    value: Any,
    *,
    scenario_id: str,
) -> dict[str, Any]:
    """Validate one object-root ``frozen_final_world_reference`` value."""

    if not isinstance(value, dict):
        raise ValueError("frozen final-world reference must be an object")
    required = {
        "schema_version",
        "scenario_id",
        "final_world_sha256",
        "final_world_root_keys",
        "source_kind",
    }
    if set(value) != required:
        raise ValueError("frozen final-world reference fields are invalid")
    if value.get("schema_version") != FROZEN_REFERENCE_VERSION:
        raise ValueError("frozen final-world reference version mismatch")
    if value.get("scenario_id") != scenario_id:
        raise ValueError("frozen final-world reference scenario mismatch")
    if value.get("source_kind") != "runtime_package":
        raise ValueError("frozen final-world reference source kind is invalid")
    _validate_digest(value.get("final_world_sha256"))
    keys = value.get("final_world_root_keys")
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
        or len(keys) != len(set(keys))
    ):
        raise ValueError("frozen final-world reference root keys must be a list of unique non-empty strings")
    return value


def _validate_digest(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(SHA256_PREFIX)
        or len(value) != len(SHA256_PREFIX) + 64
        or any(character not in "0123456789abcdef" for character in value[len(SHA256_PREFIX) :])
    ):
        raise ValueError("final-world hash must be a lowercase sha256 digest")
    return value
