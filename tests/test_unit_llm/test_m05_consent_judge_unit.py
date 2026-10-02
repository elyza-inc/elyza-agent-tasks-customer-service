"""Unit checks for M05 consent judging with deterministic transports."""

from __future__ import annotations

import json

import pytest

from elyza_agent_tasks_customer_service.evaluation.llm import m05_consent_judge as judge


def _package() -> dict:
    return {"scenario_id": "s", "contracts": {"completion": {"required_consent": "terms_agreement"}, "m05": {"deadline_tool_id": "save"}}, "inputs": {"declared": [{"name": "terms_agreement", "value": "同意します", "spoken": "了承します"}]}}


def _config() -> dict:
    return {"max_input_bytes": 10_000, "judge": {"member_id": "j", "family": "f", "provider": "test", "endpoint": "https://judge", "transport": "chat_completions", "resolved_model_revision": "model-v1", "revision_policy": "fixed", "api_key_env": "M05_TEST_KEY", "sampling": {"temperature": 0, "seed": 1, "max_tokens": 20}}}


def test_declarations_pairs_and_missing_measurement_paths() -> None:
    package = _package()
    assert judge.semantic_consent_declarations(package)[0]["field"] == "terms_agreement"
    pairs, reason = judge.build_consent_pairs(package, {"conversation": [{"actor": "customer", "content": "了承します"}, {"tool_id": "save", "result": {"ok": True}}]})
    assert reason is None and pairs[0]["evidence_ref"] == "conversation[0000]"
    pairs, reason = judge.build_consent_pairs(package, {"conversation": []})
    assert pairs == [] and reason == "m05_protected_action_not_observed"
    package["contracts"]["m05"] = None
    assert judge.judge_m05_consent(package=package, record={"conversation": []}, config={}, cache_dir=__import__("pathlib").Path("."))["status"] == "N/A"


def test_judge_success_cache_and_unavailable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("M05_TEST_KEY", "secret")
    package = _package()
    pairs, _ = judge.build_consent_pairs(package, {"conversation": [{"actor": "customer", "content": "了承します"}, {"tool_id": "save", "result": {"ok": True}}]})
    calls = 0
    def transport(_url, payload, _headers, _timeout):
        nonlocal calls
        calls += 1
        rows = [{"question_id": row["question_id"], "answer": "yes", "reason": "same"} for row in json.loads(payload["messages"][1]["content"].removeprefix("DATA_JSON="))["pairs"]]
        return {"model": "model-v1", "choices": [{"message": {"content": json.dumps({"judgements": rows})}}]}
    first = judge.judge_consent_pairs(pairs=pairs, config=_config(), cache_dir=tmp_path, transport=transport)
    second = judge.judge_consent_pairs(pairs=pairs, config=_config(), cache_dir=tmp_path, transport=transport)
    assert first["status"] == "measured" and second["judge"]["cache_reused"] and calls == 1
    monkeypatch.delenv("M05_TEST_KEY")
    unavailable = judge.judge_consent_pairs(pairs=[{**pairs[0], "question_id": "other"}], config=_config(), cache_dir=tmp_path / "missing")
    assert unavailable["status"] == "N/M" and unavailable["reason"] == "semantic_consent_judge_unavailable"


def test_pair_and_response_validation_reject_bad_order() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        judge._validate_pairs([])
    schema = judge._response_schema(["a"])
    with pytest.raises(ValueError, match="order or coverage"):
        judge._validate_response({"judgements": [{"question_id": "a", "answer": "yes", "reason": "x"}]}, [{"question_id": "b"}], schema)


def test_m05_boundary_validation_and_pass_status(monkeypatch, tmp_path) -> None:
    package = _package()
    for invalid in (
        {**package, "contracts": []},
        {**package, "inputs": []},
    ):
        with pytest.raises(ValueError, match="contracts and package.inputs"):
            judge.semantic_consent_declarations(invalid)
    for record in (
        {"conversation": {}},
        {"conversation": ["not an object"]},
    ):
        with pytest.raises(ValueError, match="conversation"):
            judge.build_consent_pairs(package, record)
    invalid_deadline = _package()
    invalid_deadline["contracts"]["m05"]["deadline_tool_id"] = ""
    with pytest.raises(ValueError, match="deadline"):
        judge.build_consent_pairs(invalid_deadline, {"conversation": []})

    pairs = [{"question_id": "q", "question": judge.QUESTION_TEXT, "declared_value": "same", "customer_utterance": "same", "evidence_ref": "e"}]
    for bad in ({**pairs[0], "question": ""}, {**pairs[0], "evidence_ref": 1}):
        with pytest.raises(ValueError, match="values"):
            judge._validate_pairs([bad])
    bad_config = _config()
    bad_config["judge"]["sampling"]["seed"] = True
    with pytest.raises(ValueError, match="seed"):
        judge._judge_member(bad_config)
    too_small = _config()
    too_small["max_input_bytes"] = False
    with pytest.raises(ValueError, match="max_input_bytes"):
        judge.judge_consent_pairs(pairs=pairs, config=too_small, cache_dir=tmp_path)

    monkeypatch.setenv("M05_TEST_KEY", "secret")
    measured = judge.judge_m05_consent(
        package=package,
        record={"conversation": [{"actor": "customer", "content": "了承します"}, {"tool_id": "save", "result": {"ok": True}}]},
        config=_config(),
        cache_dir=tmp_path,
        transport=lambda _url, payload, _headers, _timeout: {"model": "model-v1", "choices": [{"message": {"content": json.dumps({"judgements": [{"question_id": json.loads(payload["messages"][1]["content"].removeprefix("DATA_JSON="))["pairs"][0]["question_id"], "answer": "yes", "reason": "same"}]})}}]},
    )
    assert measured["status"] == "pass"
    assert measured["reason"] == "semantically_matching_consent_observed"
