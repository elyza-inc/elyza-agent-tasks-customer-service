"""Small synthetic tickets exercise direct post-call scoring branches."""

from __future__ import annotations

import json

import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring import post_call_ticket_scoring as ticket_score
from elyza_agent_tasks_customer_service.evaluation.scoring import ticket_conversation_questions as questions


def _contract():
    return {"schema_version": "post_call_ticket_contract", "comparison_fields": list(ticket_score.DETERMINISTIC_TICKET_FIELDS),
            "enum_catalog": {"handling_types": ["done"], "action_codes": ["act"], "id_types": ["id"], "evidence_kinds": ["message"]}}


def _ticket(**extra):
    value = {"handling_type": "done", "target_ids": [], "performed_actions": [], "evidence_refs": []}
    value.update(extra)
    return value


def test_template_loader_and_builder_validation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="must be JSON"):
        questions.load_ticket_question_templates(tmp_path / "x.yaml")
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(ValueError, match="cannot load"):
        questions.load_ticket_question_templates(bad)
    for value, message in [({}, "root fields"), ({"schema_version": "bad", "templates": []}, "unsupported"),
                           ({"schema_version": questions.TEMPLATE_VERSION, "templates": {}}, "must be an array")]:
        path = tmp_path / "shape.json"
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match=message):
            questions.load_ticket_question_templates(path)
    with pytest.raises(ValueError, match="ticket must be an object"):
        questions.build_ticket_conversation_questions([])
    with pytest.raises(ValueError, match="candidate_span_refs"):
        questions.build_ticket_conversation_questions({}, candidate_span_refs=[""])
    with pytest.raises(ValueError, match="code_descriptions"):
        questions.build_ticket_conversation_questions({}, code_descriptions={"": "x"})
    compiled = questions.build_ticket_conversation_questions(
        {"refusal_reason": "r", "promises": [{"owner_id": "o", "channel": "c", "deadline": "d"}],
         "answer_summary": {"diagnostic_conclusion_code": "code"}},
        candidate_span_refs=["e1"], code_descriptions={"r": "reason", "code": "summary"})
    assert [row["question_id"] for row in compiled["questions"]] == [
        "ticket:refusal_reason_code", "ticket:promises[0]:callback", "ticket:answer_summary"]
    assert "summary" in compiled["questions"][-1]["question_text"]
    with pytest.raises(ValueError, match="must be an array"):
        questions.build_ticket_conversation_questions({"promises": {"bad": True}})
    with pytest.raises(ValueError, match="no question-ready"):
        questions.build_ticket_conversation_questions({"answer_summary": {}})


def test_question_renderer_and_scalar_errors():
    with pytest.raises(ValueError, match="parameters missing"):
        questions._question(question_id="q", template={"required_parameters": ["x"], "question_text": "{x}", "template_id": "t"}, parameters={}, source_ref="s", candidate_span_refs=[])
    with pytest.raises(ValueError, match="rendering failed"):
        questions._question(question_id="q", template={"required_parameters": [], "question_text": "{x}", "template_id": "t"}, parameters={}, source_ref="s", candidate_span_refs=[])
    with pytest.raises(ValueError, match="non-empty string"):
        questions._require_scalar(None, "x")


def test_m15_questions_mix_three_yes_and_two_no(monkeypatch):
    monkeypatch.setattr(questions, "_source_domain_disclosures", lambda domain_id, scenario_id: ["別シナリオの必須案内"])
    kwargs = dict(
        ticket={"answer_summary": "回答要約"},
        scenario_id="scenario-a",
        completion={
            "required_disclosures": ["必須案内A", "必須案内B"],
            "required_consent": "同意案内",
            "forbidden_mutations": [{"table": "reservations", "set": {"status": "cancelled"}}],
        },
        domain_id="domain",
    )
    compiled = questions.build_ticket_conversation_questions(**kwargs)

    assert len(compiled["questions"]) == 5
    assert [row["expected_answer"] for row in compiled["questions"]] == [
        "yes", "yes", "yes", "no", "no"
    ]
    assert questions.build_ticket_conversation_questions(**kwargs) == compiled


def test_m15_decoy_excludes_a_correct_disclosure_overlap(monkeypatch):
    monkeypatch.setattr(questions, "_source_domain_disclosures", lambda domain_id, scenario_id: ["重複する案内", "別の案内"])
    compiled = questions.build_ticket_conversation_questions(
        {},
        scenario_id="scenario-b",
        completion={
            "required_disclosures": ["重複する案内"],
            "required_consent": None,
            "forbidden_mutations": [{"table": "cases", "set": {}}],
        },
        domain_id="domain",
    )

    decoy = next(row for row in compiled["questions"] if row["template_id"] == "peer_required_disclosure")
    assert "別の案内" in decoy["question_text"]
    assert "重複する案内" not in decoy["question_text"]


def test_compare_and_deterministic_post_call_branches(monkeypatch):
    expected = _ticket()
    assert ticket_score.compare_ticket_facts(expected=expected, observed=expected, target_candidates=[])["score"] == 1.0
    target_ids = [
        {"id_type": "id", "value": "first"},
        {"id_type": "id", "value": "second"},
        {"id_type": "id", "value": "first"},
    ]
    reordered = [target_ids[2], target_ids[0], target_ids[1]]
    assert ticket_score.compare_ticket_facts(
        expected=_ticket(target_ids=target_ids), observed=_ticket(target_ids=reordered), target_candidates=[]
    )["field_results"]["target_ids"]
    different_targets = [target_ids[0], target_ids[1], target_ids[1]]
    assert not ticket_score.compare_ticket_facts(
        expected=_ticket(target_ids=target_ids), observed=_ticket(target_ids=different_targets), target_candidates=[]
    )["field_results"]["target_ids"]
    mismatch = ticket_score.compare_ticket_facts(expected=expected, observed=_ticket(handling_type="no"), target_candidates=[])
    assert mismatch["score"] == .75 and mismatch["mismatches"][0]["field"] == "handling_type"
    for artifact, status, reason in [
        (None, "not_measurable", "operator_ticket_missing"),
        ({"status": "infra_error", "reason": "down"}, "not_measurable", "down"),
        ({"status": "contract_invalid"}, "contract_invalid", "ticket_artifact_contract_invalid"),
        ({"status": "bad"}, "fail", "operator_ticket_invalid"),
    ]:
        row = ticket_score.score_post_call_ticket(contract=_contract(), artifact=artifact, initial_world={}, final_world={}, call_events=[])
        assert row["status"] == status and reason in row["reason"]
    monkeypatch.setattr(ticket_score, "build_ticket_fact", lambda *args, **kwargs: expected)
    row = ticket_score.score_post_call_ticket(contract=_contract(), artifact={"status": "submitted", "ticket": expected}, initial_world={}, final_world={}, call_events=[])
    assert row["status"] == "pass" and row["metric_results"]["M15"]["ticket_fidelity_conversational"] == "N/A"


@pytest.mark.parametrize("candidate, message", [
    ({}, "fields are invalid"),
    ({"schema_version": "bad", "enum_catalog": {k: ["x"] for k in ticket_score.DETERMINISTIC_ENUM_CATALOGS}, "comparison_fields": list(ticket_score.DETERMINISTIC_TICKET_FIELDS)}, "unsupported"),
])
def test_deterministic_contract_errors(candidate, message):
    with pytest.raises(ValueError, match=message):
        ticket_score._validate_deterministic_contract(candidate)
    with pytest.raises(ValueError, match="ticket fields"):
        ticket_score._validate_deterministic_ticket({}, _contract()["enum_catalog"])


@pytest.mark.parametrize(
    ("ticket", "message"),
    [(_ticket(handling_type="bad"), "handling_type"),
     (_ticket(target_ids=[{}]), "target_ids\\[0\\] fields"),
     (_ticket(target_ids=[{"id_type": "id", "value": ""}]), "non-empty"),
     (_ticket(performed_actions=[{}]), "performed_actions\\[0\\] fields"),
     (_ticket(performed_actions=[{"action_code": "act", "target_ref": 1, "outcome": "succeeded", "receipt_refs": []}]), "string or null"),
     (_ticket(performed_actions=[{"action_code": "act", "target_ref": None, "outcome": "wat", "receipt_refs": []}]), "outcome"),
     (_ticket(performed_actions=[{"action_code": "act", "target_ref": None, "outcome": "succeeded", "receipt_refs": [1]}]), "string array"),
     (_ticket(evidence_refs=[{}]), "evidence_refs\\[0\\] fields"),
     (_ticket(evidence_refs=[{"kind": "message", "ref": ""}]), "non-empty")],
)
def test_deterministic_ticket_validation_errors(ticket, message):
    with pytest.raises((ValueError, ticket_score.TicketEnumSelectionError), match=message):
        ticket_score._validate_deterministic_ticket(ticket, _contract()["enum_catalog"])


