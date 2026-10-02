"""Regression checks for D5 deadline selection and D6 M04 prerequisites."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts.artifact_receipts import (
    canonical_json_bytes,
)
from elyza_agent_tasks_customer_service.evaluation.llm.m05_consent_judge import (
    build_consent_pairs,
    judge_consent_pairs,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.conversation_log_scoring import (
    _evidence,
    _run_judge,
    _tasks,
    _validate_evidence_rules,
    load_judge_config,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.core_metric_scoring import (
    score_m04_customer_trigger_control,
)


ROOT = Path(__file__).resolve().parents[1]
# 2026-09-28 readback_role 導入・M16-M2 基準の明記・ジャッジの luna 切り替えで request が変わったため更新
# Updated 2026-09-30 (b5d2e1a): the judge prompt now documents the M15 kinds and the
# required_consent evidence rule, so every judge request changed by design.
LEGACY_NO_INJECTION_DISCLOSURE_REQUEST_SHA256 = (
    "a0d0de1c1fa9f68a0d3bb445702a48bc33f8b36f0a7dfbc4d6d68edd54315cb5"
)
LEGACY_NO_INJECTION_CONSENT_REQUEST_SHA256 = (
    "7fe097ad9c64293e4dc6282cfd2a013e4a18351135936780f0b4aaaa5472ff2f"
)


@pytest.fixture(autouse=True)
def _declared_m05_bound_without_gold(monkeypatch):
    """Synthetic packages here have no gold dialogue; their declared M05 deadline Tool is the bound."""

    from elyza_agent_tasks_customer_service.evaluation.scoring import conversation_log_scoring as judge

    original = judge.m05_judge_deadline_tool
    monkeypatch.setattr(
        judge,
        "m05_judge_deadline_tool",
        lambda package, scenario_id: original(package, scenario_id)
        if "gold_dialogue" in package
        else package["contracts"]["m05"]["deadline_tool_id"],
    )


def _d5_package() -> dict:
    return {
        "scenario_id": "d5-request-stability",
        "contracts": {
            "completion": {"required_consent": "terms_agreement"},
            "m05": {
                "disclosure": "手数料が発生します",
                "deadline_tool_id": "update_booking",
            },
            "m20": None,
        },
        "inputs": {
            "declared": [
                {
                    "name": "terms_agreement",
                    "value": "同意します",
                    "spoken": "了承します",
                }
            ],
        },
        "persona": {},
    }


def _d5_record(*, injected: bool) -> dict:
    if not injected:
        return {
            "conversation": [
                {"actor": "customer", "content": "予約を変更したいです", "seq": 1},
                {"actor": "operator", "content": "手数料が発生します", "seq": 2},
                {"actor": "customer", "content": "了承します", "seq": 3},
                {
                    "actor": "operator",
                    "tool_id": "update_booking",
                    "arguments": {},
                    "result": {"ok": True},
                    "seq": 4,
                },
            ],
            "tool_calls": [
                {
                    "tool_id": "update_booking",
                    "arguments": {},
                    "result": {"ok": True},
                    "seq": 4,
                }
            ],
            "event_log": [],
        }
    failed = {
        "actor": "operator",
        "tool_id": "update_booking",
        "arguments": {},
        "result": {"ok": False, "error": "injected"},
        "seq": 2,
    }
    succeeded = {
        "actor": "operator",
        "tool_id": "update_booking",
        "arguments": {},
        "result": {"ok": True},
        "seq": 5,
    }
    return {
        "conversation": [
            {"actor": "customer", "content": "予約を変更したいです", "seq": 1},
            failed,
            {"actor": "operator", "content": "手数料が発生します", "seq": 3},
            {"actor": "customer", "content": "了承します", "seq": 4},
            succeeded,
        ],
        "tool_calls": [failed, succeeded],
        "event_log": [],
    }


def _m05_task(package: dict, record: dict) -> tuple[dict, dict]:
    evidence = _evidence(record)
    tasks, _ = _tasks(package, record, evidence)
    return next(task for task in tasks if task["metric_id"] == "M05"), evidence


def test_d5_uses_first_successful_deadline_call() -> None:
    package = _d5_package()
    record = _d5_record(injected=True)

    task, evidence = _m05_task(package, record)
    context = task["relevant_context"]
    sources = {
        row["ref"]: row
        for row in evidence["conversation"] + evidence["tool_calls"]
    }
    pairs, reason = build_consent_pairs(package, record)

    assert context["deadline_ref"] == "tool_calls[0001]"
    assert context["deadline_seq"] == 5
    assert _validate_evidence_rules(
        task,
        "yes",
        ["conversation[0002]"],
        sources,
    ) == []
    assert reason is None
    assert [pair["evidence_ref"] for pair in pairs] == [
        "conversation[0000]",
        "conversation[0003]",
    ]


def test_d5_falls_back_to_first_call_without_success() -> None:
    package = _d5_package()
    record = _d5_record(injected=True)
    record["conversation"][-1]["result"] = {"ok": False, "error": "again"}
    record["tool_calls"][-1]["result"] = {"ok": False, "error": "again"}

    task, _ = _m05_task(package, record)
    pairs, reason = build_consent_pairs(package, record)

    assert task["relevant_context"]["deadline_ref"] == "tool_calls[0000]"
    assert task["relevant_context"]["deadline_seq"] == 2
    assert reason is None
    assert [pair["evidence_ref"] for pair in pairs] == ["conversation[0000]"]


def test_d5_non_injection_judge_requests_are_byte_identical(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = _d5_package()
    record = _d5_record(injected=False)
    task, evidence = _m05_task(package, record)
    pairs, reason = build_consent_pairs(package, record)
    config = load_judge_config(ROOT / "configs" / "conversation_log_judge.yaml")
    monkeypatch.setenv(config["judge"]["api_key_env"], "test")
    disclosure_requests = []
    consent_requests = []

    def disclosure_transport(url, payload, headers, timeout):
        disclosure_requests.append(payload)
        content = {
            "judgements": [
                {"question_id": task["question_id"], "answer": "no", "evidence": []}
            ]
        }
        return {
            "model": config["judge"]["resolved_model_revision"],
            "choices": [{"message": {"content": json.dumps(content)}}],
        }

    def consent_transport(url, payload, headers, timeout):
        consent_requests.append(payload)
        content = {
            "judgements": [
                {
                    "question_id": pair["question_id"],
                    "answer": "no",
                    "reason": "test",
                }
                for pair in pairs
            ]
        }
        return {
            "model": config["judge"]["resolved_model_revision"],
            "choices": [{"message": {"content": json.dumps(content)}}],
        }

    assert reason is None
    _run_judge(
        tasks=[task],
        evidence=evidence,
        config=config,
        cache_dir=tmp_path / "disclosure",
        transport=disclosure_transport,
    )
    judge_consent_pairs(
        pairs=pairs,
        config=config,
        cache_dir=tmp_path / "consent",
        transport=consent_transport,
    )

    disclosure_hash = hashlib.sha256(
        canonical_json_bytes(disclosure_requests[0])
    ).hexdigest()
    consent_hash = hashlib.sha256(canonical_json_bytes(consent_requests[0])).hexdigest()
    assert disclosure_hash == LEGACY_NO_INJECTION_DISCLOSURE_REQUEST_SHA256
    assert consent_hash == LEGACY_NO_INJECTION_CONSENT_REQUEST_SHA256


def _m04_contract() -> dict:
    return {
        "schema_version": "m04_customer_trigger_contract",
        "edges": [
            {
                "edge_id": "identity-before-update",
                "customer_trigger_type": "customer_pressure",
                "critical_action_tool_id": "update_booking",
                "required_precondition_tool_ids": ["verify_identity"],
            }
        ],
    }


def test_d6_precondition_before_trigger_passes() -> None:
    score = score_m04_customer_trigger_control(
        contract=_m04_contract(),
        call_events=[
            {"event_type": "tool_call", "tool": "verify_identity"},
            {
                "actor": "user",
                "event_type": "message",
                "persona_fired": "customer_pressure",
            },
            {"event_type": "tool_call", "tool": "update_booking"},
        ],
    )

    assert score["status"] == "pass"
    assert score["value"]["edges"][0]["precondition_failures"] == []


def test_d6_missing_precondition_still_fails() -> None:
    score = score_m04_customer_trigger_control(
        contract=_m04_contract(),
        call_events=[
            {
                "actor": "user",
                "event_type": "message",
                "persona_fired": "customer_pressure",
            },
            {"event_type": "tool_call", "tool": "update_booking"},
        ],
    )

    assert score["status"] == "fail"
    assert score["value"]["edges"][0]["precondition_failures"] == [
        {
            "critical_action_position": 1,
            "missing_or_late_precondition_tool_ids": ["verify_identity"],
        }
    ]


def test_d6_critical_action_can_satisfy_its_own_declared_requirement() -> None:
    contract = _m04_contract()
    contract["edges"][0]["required_precondition_tool_ids"].append("update_booking")

    score = score_m04_customer_trigger_control(
        contract=contract,
        call_events=[
            {
                "actor": "user",
                "event_type": "message",
                "persona_fired": "customer_pressure",
            },
            {"event_type": "tool_call", "tool": "verify_identity"},
            {"event_type": "tool_call", "tool": "update_booking"},
        ],
    )

    assert score["status"] == "pass"
    assert score["value"]["edges"][0]["precondition_failures"] == []
