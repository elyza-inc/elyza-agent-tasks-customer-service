"""Post-call ticket field and action-code constants, and the canonical world hash used by the package adapter."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any


ROW_IDENTITY_FIELDS = ("role", "role_name", "customer", "sequence")
TICKET_COMPARISON_FIELDS = (
    "handling_type",
    "target_ids",
    "performed_actions",
    "evidence_refs",
)
TICKET_CATALOG_ACTION_CODES = (
    "search_sop_categories",
    "search_sops_in_category",
    "get_sop",
)


def _world_hash(world: dict[str, list[dict[str, Any]]], scenario_id: str) -> str:
    """Hash a table mapping with the production row ordering contract."""

    if not isinstance(world, dict) or not all(isinstance(rows, list) for rows in world.values()):
        raise ValueError("world must be a table-to-row-list mapping")
    canonical = {}
    for table, rows in sorted(world.items()):
        for row in rows:
            identity = row.get("row_identity")
            if not isinstance(identity, dict) or any(identity.get(field) is None for field in ROW_IDENTITY_FIELDS):
                raise ValueError(f"{scenario_id}: world.row_identity: required fields are missing")
        canonical[table] = sorted(
            (deepcopy(row) for row in rows),
            key=lambda row: tuple(str(row["row_identity"][field]) for field in ROW_IDENTITY_FIELDS),
        )
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()
