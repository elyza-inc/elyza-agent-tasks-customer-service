"""Small observable boundaries for the remaining sampled scoring mutants."""

from __future__ import annotations

import json

import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring import package_scoring as package
from elyza_agent_tasks_customer_service.evaluation.scoring import ticket_conversation_questions as questions


def test_ticket_question_template_and_summary_boundaries(tmp_path):
    templates = [{"template_id": item, "question_text": "q", "required_parameters": []} for item in questions.TEMPLATE_IDS]
    for bad in ({**templates[0], "extra": 1}, {**templates[0], "question_text": ""}, {**templates[0], "required_parameters": [""]}):
        rows = [bad, *templates[1:]]
        path = tmp_path / "templates.json"
        path.write_text(json.dumps({"schema_version": questions.TEMPLATE_VERSION, "templates": rows}))
        with pytest.raises(ValueError):
            questions.load_ticket_question_templates(path)
    assert questions._answer_summary_text({"summary": "", "diagnostic_conclusion_code": "code"}, {"code": "description"}) == "description"
    with pytest.raises(ValueError, match="non-empty string"):
        questions._require_scalar(1, "refusal_reason_code")


def test_package_merge_filters_and_nonfail_consent():
    core_report = {"metric_results": [{"metric_id": "M14", "metric_instance_id": "m14", "status": "pass"}, {"metric_id": "M01", "metric_instance_id": "m01", "status": "pass"}], "violations": [{"metric_id": "M01"}, {"metric_id": "M14"}]}
    ticket_score = {"status": "pass", "reason": "", "schema_version": "ticket", "metric_results": {"M14": {"passed": True}}}
    package._merge_ticket_row(core_report, ticket_score)
    assert core_report["metric_results"][0]["metric_instance_id"] == "m14"
    assert core_report["metric_results"][0]["reason"] == "pass"
    assert core_report["violations"] == [{"metric_id": "M01"}]
    m05 = {"metric_results": [{"metric_id": "M05", "metric_instance_id": "m05", "status": "pass", "violations": [], "diagnostics": {}}], "violations": []}
    package._merge_m05_consent(m05, {"status": "N/M", "reason": "missing", "schema_version": "s", "judgements": [], "judge": {}})
    assert m05["metric_results"][0]["status"] == "N/M" and not m05["violations"]
