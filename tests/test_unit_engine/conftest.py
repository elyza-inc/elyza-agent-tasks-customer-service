"""Fixtures for isolated engine-runtime tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def htl_package() -> dict:
    """Return one assembled package with real lock-recovery declarations."""

    return json.loads((ROOT / ".run" / "packages" / "htl-001.yaml").read_text())


@pytest.fixture
def minimal_package() -> dict:
    """Return a smallest valid runtime package with one customer and item table."""

    return {
        "scenario_id": "engine-unit",
        "business": {"initial_user_request": "状態を変更したい"},
        "inputs": {"identity": [], "declared": [], "reference_date_spec": {}},
        "domain": {},
        "world_schema": {
            "customer_table": "customers",
            "search_policy": {"order_by": "row_identity", "default_limit": 10, "max_limit": 10},
            "entities": {"items": {"columns": {"key": "string", "status": "string", "count": "integer"}}},
        },
        "world": {
            "customers": [{"row_identity": {"customer": "c1"}, "values": {"name": "ナマエ"}}],
            "items": [{"row_identity": {"customer": "c1", "sequence": 1}, "values": {"key": "one", "status": "old", "count": 1}}],
        },
        "tools": {
            "verify": {"id": "verify", "operation": "filtered_search", "entity": "customers", "description": "本人確認", "returns": "顧客", "arguments": [{"name": "name", "type": "string", "required": True}], "filters": [{"column": "name", "argument": "name", "operator": "eq"}]},
            "search": {"id": "search", "operation": "filtered_search", "entity": "items", "description": "検索", "returns": "一覧", "arguments": [{"name": "key", "type": "string", "required": True}], "filters": [{"column": "key", "argument": "key", "operator": "eq"}]},
            "update": {"id": "update", "operation": "update", "entity": "items", "description": "更新", "returns": "更新行", "arguments": [{"name": "key", "type": "string", "required": True}, {"name": "status", "type": "string", "required": True}], "filters": [{"column": "key", "argument": "key", "operator": "eq"}], "mutation": {"set": {"status": {"from_argument": "status"}}}},
            "create": {"id": "create", "operation": "create", "entity": "items", "description": "追加", "returns": "追加行", "arguments": [{"name": "key", "type": "string", "required": True}, {"name": "status", "type": "string", "required": True}], "filters": [], "mutation": {"set": {"key": {"from_argument": "key"}, "status": {"from_argument": "status"}}}},
        },
        "sop_catalog": {"categories": {"basic": {"title": "基本", "sops": [{"id": "s1", "title": "手順", "applicability": "全件", "completion": "完了", "prohibition": "なし", "escalation": "なし", "steps": [{"step_id": "q", "kind": "question", "description": "確認"}]}]}}},
        "persona": {},
    }


@pytest.fixture
def duplicate_minimal_package(minimal_package: dict) -> dict:
    """Return the synthetic package with two rows matching the same completion key."""

    package = deepcopy(minimal_package)
    package["world"]["items"].append(
        {"row_identity": {"customer": "c1", "sequence": 2}, "values": {"key": "two", "status": "old", "count": 2}}
    )
    return package
