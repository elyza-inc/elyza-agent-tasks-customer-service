from __future__ import annotations

import json

import pytest

from elyza_agent_tasks_customer_service.evaluation.observers import (
    japanese_parse_adapter as parser,
    japanese_rule_observer as rules,
)


def test_parser_lock_hashes_and_dependency_nm(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "tree"
    source.mkdir()
    (source / "x.txt").write_text("x", encoding="utf-8")
    assert parser._hash_tree(source).startswith("sha256:")
    assert parser._hash_file(source / "x.txt").startswith("sha256:")
    parser._validate_turns([{"turn_ref": "o1", "text": "確認します"}])
    with pytest.raises(ValueError, match="unique"):
        parser._validate_turns([{"turn_ref": "o1", "text": "x"}, {"turn_ref": "o1", "text": "y"}])
    monkeypatch.setattr(parser, "parse_japanese_turns", lambda **_: (_ for _ in ()).throw(parser.JapaneseParserDependencyError("missing")))
    assert parser.parse_japanese_turns_or_nm(turns=[{"turn_ref": "o1", "text": "x"}])["status"] == "N/M"
    assert parser._is_known_dependency_startup_error(ModuleNotFoundError("x", name="ginza"))


def test_prefixed_in_group_honorific_is_customer_side_not_a_violation() -> None:
    """Treat ご担当者様 as a customer-side reference without a residual candidate."""

    context = {
        "persons": [
            {
                "person_id": "staff",
                "relation_to_operator": "in_group",
                "surface_forms": ["担当者"],
            }
        ]
    }
    family = rules._in_out_group_family(
        {"turns": [{"turn_ref": "operator:1", "text": "ご担当者様におつなぎします。"}]},
        context,
        rules.load_rule_catalog(),
    )

    assert family["parser_measurable"] is True
    assert family["violations"] == []
    assert family["diagnostics"]["unresolved_occurrences"] == []


def test_operator_side_modifier_makes_an_ambiguous_staff_surface_a_violation() -> None:
    context = {
        "persons": [
            {"person_id": "staff-one", "relation_to_operator": "in_group", "surface_forms": ["担当者"]},
            {"person_id": "staff-two", "relation_to_operator": "in_group", "surface_forms": ["担当者"]},
        ]
    }
    family = rules._in_out_group_family(
        {"turns": [{"turn_ref": "operator:1", "text": "当社の担当者様が対応します。"}]},
        context,
        rules.load_rule_catalog(),
    )

    assert family["violations"][0]["reason_code"] == "in_group_person_raised_to_external_customer"
    assert family["diagnostics"]["unresolved_occurrences"] == []


def test_parser_and_rule_validation_edges(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = parser.load_parser_lock()
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    assert parser.load_parser_lock(path)["model_name"] == lock["model_name"]
    with pytest.raises(ValueError, match="turns"):
        parser._validate_turns({})
    monkeypatch.setattr(parser, "_load_ginza", lambda *_: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError, match="boom"):
        parser.parse_japanese_turns(turns=[{"turn_ref": "x", "text": "x"}])
    with pytest.raises(ValueError, match="context"):
        rules.validate_register_context({})
