"""Boundary tests for scalar normalization and provenance matching."""

from __future__ import annotations

from math import inf

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts import type_contracts as types


@pytest.mark.parametrize(("value", "kind"), [(True, "boolean"), (1, "integer"), (1.5, "number"), ("x", "string"), (None, "NoneType")])
def test_python_types_and_finite_normalization(value, kind):
    assert types.python_value_type(value) == kind
    assert types.value_matches_type(value, kind) == (kind != "NoneType")
    if kind == "number":
        assert types.normalize_typed_value(value, kind) == value


def test_normalizers_and_invalid_scalar_errors():
    assert types._normalized_date("２０２６年2月3日") == "2026-02-03"
    assert types._normalized_date("2026-02-30") is None
    assert types._normalized_match_value("０９０ー１２３４ ５６７８", "phone", None) == "09012345678"
    assert types._normalized_match_value(" カ ナ ", "customer_kana", None) == "カナ"
    assert types._normalized_match_value(1, "x", "number") == 1.0
    assert types._normalized_match_value(None, "x", None) is None
    assert types._normalized_match_value(" 2026-02-03 ", "x_date", "date") == "2026-02-03"
    assert types._japanese_number_forms(60) == set()
    assert types._japanese_integer("一万二百三") == 10203
    with pytest.raises(ValueError, match="expected integer, got boolean"):
        types.normalize_typed_value(True, "integer")
    assert not types.value_matches_type(inf, "number")


def test_schema_loader_errors(monkeypatch):
    monkeypatch.setattr(types.yaml, "safe_load", lambda text: {})
    with pytest.raises(RuntimeError, match="string list"):
        types._load_scalar_types()
    monkeypatch.setattr(types.yaml, "safe_load", lambda text: {"definitions": {"scalar_type_values": ["x", "x"]}})
    with pytest.raises(RuntimeError, match="duplicates"):
        types._load_scalar_types()


@pytest.mark.parametrize(
    ("candidate", "expected", "contract"),
    [
        ("午後十時です", "22:00", [("arrival_time", "time")]),
        ("5月2日", "2026-05-02", [("date", "date")]),
        ("2026年5月2日 14時30分", "2026-05-02 14:30", [("when", "string")]),
        ("五十人", 50, [("capacity", "integer")]),
        ("ゼロキュウゼロ-イチニサンヨン-ゴロクナナハチ", "09012345678", [("phone_number", "string")]),
        ("素泊まりで", "素泊まり山風", [("plan_name", "string")]),
    ],
)
def test_provenance_variants(candidate, expected, contract):
    assert types.provenance_text_contains_typed_value(candidate, expected, contract)


def test_provenance_rejects_invalid_and_zero_substrings():
    assert not types.provenance_text_contains_typed_value(1, "x", [("x", "string")])
    assert not types._provenance_text_contains_number("001", 0)
    assert not types._provenance_text_contains_number("x", True)
    assert not types._provenance_text_contains_number("x", inf)
    assert not types._provenance_text_contains_number("x", .5)
    assert not types._provenance_text_contains_number("x", 3)
    assert not types._provenance_text_contains_time("22時", "25:00")
    assert not types._provenance_text_contains_date("5月2日", "bad")
    assert not types.provenance_text_contains_declared_spoken("x", "")
    assert not types.provenance_text_contains_typed_value("abc", 1, [("x", "string")])
    assert not types.provenance_text_contains_typed_value("abc", "z", [("x", "string")])
    assert types.is_operational_enum_value("reissue_count", 0)
    assert not types.is_operational_enum_value("reissue_count", False)
