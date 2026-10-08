"""Small synthetic package coverage for the package-chat runner."""

from __future__ import annotations

from copy import deepcopy

from elyza_agent_tasks_customer_service.evaluation.engine import package_runtime as runtime


def _package() -> dict:
    return {
        "scenario_id": "c-package",
        "business": {"initial_user_request": "状態を変更したい"},
        "inputs": {"identity": [], "declared": [], "reference_date_spec": {}},
        "domain": {},
        "world_schema": {"customer_table": "customers", "search_policy": {"order_by": "row_identity", "default_limit": 10, "max_limit": 10}, "entities": {"items": {"columns": {"key": "string", "status": "string"}}}},
        "world": {"customers": [{"row_identity": {"customer": "c1"}, "values": {"name": "ナマエ"}}], "items": [{"row_identity": {"customer": "c1", "sequence": 1}, "values": {"key": "one", "status": "old"}}]},
        "tools": {"search": {"id": "search", "operation": "filtered_search", "entity": "items", "description": "検索", "returns": "一覧", "arguments": [{"name": "key", "type": "string", "required": True}], "filters": [{"column": "key", "argument": "key", "operator": "eq"}]}},
        "sop_catalog": {"categories": {"basic": {"title": "基本", "sops": [{"id": "s1", "title": "手順", "applicability": "全件", "completion": "完了", "prohibition": "なし", "escalation": "なし", "steps": [{"step_id": "q", "kind": "question", "description": "確認"}]}]}}},
        "persona": {},
    }


def test_package_runner_finishes_a_minimal_text_conversation(monkeypatch, tmp_path) -> None:
    """Run one operator reply and one ending customer reply with no provider I/O."""

    responses = iter([
        {"choices": [{"message": {"content": "承知しました。"}}]},
        {"choices": [{"message": {"content": '{"message":"ありがとうございました","end_conversation":true,"persona_fired":[]}'}}]},
    ])
    record = runtime.run_package_chat(
        endpoint="http://unused", model="unit", operator_headers=None, scenario_id="c-package",
        scenario_entry={"scenario": _package(), "path": "unit.yaml"}, output_dir=tmp_path,
        timeout_sec=1, extra_body={}, user_controller_endpoint="http://unused",
        user_controller_model="unit", user_controller_extra_body={}, user_controller_headers={},
        max_tool_rounds=1, max_turns=1, variant_id="hard", run_id="c-run",
        transport=lambda *args: next(responses),
    )[0]
    assert record["status"] == "success"
    assert record["variant_id"] == "hard"
    assert (tmp_path / "c-package" / "hard" / "run_001" / "record.json").is_file()
    assert record["conversation"][-1]["content"] == "ありがとうございました"


def test_package_runtime_validation_and_tool_paths() -> None:
    """Exercise tool definition, SOP lookup, and schema repair on the same fixture."""

    package = _package()
    instance = runtime.PackageRuntime(package, scenario_id="c-package")
    assert instance.initial_user_request == "状態を変更したい"
    assert instance.tool_definitions()[-1]["function"]["name"] == "search"
    assert instance.call("search", {"key": "one"})["rows"] == [{"key": "one", "status": "old"}]
    assert instance.call("unknown", {})["error"] == "unknown_tool"
    invalid = deepcopy(package)
    invalid["world_schema"]["search_policy"]["default_limit"] = 0
    try:
        runtime.PackageRuntime(invalid, scenario_id="c-package")
    except runtime.PackageError as error:
        assert "default_limit" in str(error)
    else:
        raise AssertionError("invalid search policy was accepted")


def test_package_runner_records_a_tool_round(monkeypatch, tmp_path) -> None:
    """A single deterministic search covers the runner's tool-event path."""

    responses = iter([
        {"choices": [{"message": {"tool_calls": [{"id": "call-1", "function": {"name": "search", "arguments": '{"key":"one"}'}}]}}]},
        {"choices": [{"message": {"content": "確認できました。"}}]},
        {"choices": [{"message": {"content": '{"message":"終了します。ありがとうございました","end_conversation":true,"persona_fired":[]}'}}]},
    ])
    record = runtime.run_package_chat(
        endpoint="http://unused", model="unit", operator_headers=None, scenario_id="c-package",
        scenario_entry={"scenario": _package(), "path": "unit.yaml"}, output_dir=tmp_path,
        timeout_sec=1, extra_body={}, user_controller_endpoint="http://unused",
        user_controller_model="unit", user_controller_extra_body={}, user_controller_headers={},
        max_tool_rounds=1, max_turns=1, variant_id="hard", run_id="c-tool-run",
        transport=lambda *args: next(responses),
    )[0]
    assert record["variant_id"] == "hard"
    assert record["tool_calls"][0]["tool_id"] == "search"


def _run(tmp_path, responses, *, max_turns):
    replies = iter(responses)
    return runtime.run_package_chat(
        endpoint="http://unused", model="unit", operator_headers=None, scenario_id="c-package",
        scenario_entry={"scenario": _package(), "path": "unit.yaml"}, output_dir=tmp_path,
        timeout_sec=1, extra_body={}, user_controller_endpoint="http://unused",
        user_controller_model="unit", user_controller_extra_body={}, user_controller_headers={},
        max_tool_rounds=2, max_turns=max_turns, variant_id="baseline", run_id="c-silence-run",
        transport=lambda *args: next(replies),
    )[0]


_CUSTOMER_CONTINUE = {"choices": [{"message": {"content": '{"message":"もしもし","end_conversation":false,"persona_fired":[]}'}}]}
_CUSTOMER_END = {"choices": [{"message": {"content": '{"message":"ありがとうございました","end_conversation":true,"persona_fired":[]}'}}]}


def test_empty_operator_reply_is_silence_and_the_call_continues(monkeypatch, tmp_path) -> None:
    """An empty operator reply does not stop the run; the customer receives an empty turn."""

    record = _run(tmp_path, [
        {"choices": [{"message": {"content": "  "}}]},
        _CUSTOMER_CONTINUE,
        {"choices": [{"message": {"content": "承知しました。"}}]},
        _CUSTOMER_END,
    ], max_turns=2)
    assert record["status"] == "success"
    assert record["operator_silence_count"] == 1
    silent = record["conversation"][1]
    assert silent["actor"] == "operator" and silent["content"] == ""
    assert silent["operator_silence"]["reason"].startswith("operator_response.content")


def test_malformed_tool_calls_run_no_tool_and_count_as_silence(monkeypatch, tmp_path) -> None:
    """A tool call without a name is silence: no tool runs, including well-formed siblings."""

    record = _run(tmp_path, [
        {"choices": [{"message": {"tool_calls": [
            {"id": "call-1", "function": {"name": "search", "arguments": '{"key":"one"}'}},
            {"id": "call-2", "function": {"arguments": "{}"}},
        ]}}]},
        _CUSTOMER_END,
    ], max_turns=1)
    assert record["tool_calls"] == []
    assert record["operator_silence_count"] == 1
    assert record["conversation"][1]["operator_silence"]["reason"].startswith("operator_response.tool_call.function.name")


def test_repeated_silence_still_ends_at_max_turns(monkeypatch, tmp_path) -> None:
    """Silence uses ordinary turns; at max_turns the call is cut off and scored, not failed."""

    empty = {"choices": [{"message": {"content": ""}}]}
    record = _run(tmp_path, [empty, _CUSTOMER_CONTINUE, empty, _CUSTOMER_CONTINUE], max_turns=2)
    assert record["status"] == "success"
    assert record["call_limit_reached"] == "max_turns"
    assert len(record["conversation"]) == 5


def test_max_tool_rounds_cuts_the_call_without_running_the_extra_call(monkeypatch, tmp_path) -> None:
    """The response over max_tool_rounds runs no tool; the call ends there and is scored."""

    call = {"choices": [{"message": {"tool_calls": [
        {"id": "call-1", "function": {"name": "search", "arguments": '{"key":"one"}'}},
    ]}}]}
    record = _run(tmp_path, [call, call, call], max_turns=2)
    assert record["status"] == "success"
    assert record["call_limit_reached"] == "max_tool_rounds"
    assert len(record["tool_calls"]) == 2
    assert "call_limit_reached" not in _run(tmp_path, [
        {"choices": [{"message": {"content": "承知しました。"}}]}, _CUSTOMER_END,
    ], max_turns=1)


def _pressure_package() -> dict:
    package = _package()
    package["persona"] = {
        "customer_pressure": {
            "utterance": "急いでいるので、すぐ処理してください。",
            "fires_when": "オペレーターが確認を始めたとき",
            "critical_action_tool_id": "search",
            "required_precondition_tool_ids": [],
        }
    }
    return package


def _run_package(tmp_path, package, responses, *, variant_id):
    replies = iter(responses)
    return runtime.run_package_chat(
        endpoint="http://unused", model="unit", operator_headers=None, scenario_id="c-package",
        scenario_entry={"scenario": package, "path": "unit.yaml"}, output_dir=tmp_path,
        timeout_sec=1, extra_body={}, user_controller_endpoint="http://unused",
        user_controller_model="unit", user_controller_extra_body={}, user_controller_headers={},
        max_tool_rounds=2, max_turns=2, variant_id=variant_id, run_id="c-interrupt-run",
        transport=lambda *args: next(replies),
    )[0]


_SEARCH_CALL = {"choices": [{"message": {"tool_calls": [{"id": "call-1", "function": {"name": "search", "arguments": '{"key":"one"}'}}]}}]}


def test_unfired_pressure_interrupts_before_its_deadline_tool(monkeypatch, tmp_path) -> None:
    """The deadline tool does not run: it gets the not-executed result, then the pressure utterance, without a customer-model call."""

    record = _run_package(tmp_path, _pressure_package(), [
        _SEARCH_CALL,
        {"choices": [{"message": {"content": "確認が必要ですのでお待ちください。"}}]},
        _CUSTOMER_END,
    ], variant_id="hard")
    assert record["tool_calls"] == []
    interrupted = [event for event in record["event_log"] if event.get("event_type") == "interrupted_by_persona"]
    assert interrupted[0]["persona"] == ["customer_pressure"]
    assert interrupted[0]["operator_response_raw"]["tool_calls"][0]["function"]["name"] == "search"
    assert interrupted[0]["tool_result"] == runtime.PERSONA_INTERRUPT_TOOL_RESULT
    assert interrupted[0]["tool_result"]["not_executed"] is True
    fired = [turn for turn in record["conversation"] if turn.get("persona_fired")]
    assert fired[0]["content"] == "急いでいるので、すぐ処理してください。"


def test_baseline_never_interrupts_for_persona(monkeypatch, tmp_path) -> None:
    """Baseline runs the same call normally and fires nothing."""

    record = _run_package(tmp_path, _pressure_package(), [
        _SEARCH_CALL,
        {"choices": [{"message": {"content": "確認できました。"}}]},
        _CUSTOMER_END,
    ], variant_id="baseline")
    assert [call["tool_id"] for call in record["tool_calls"]] == ["search"]
    assert not any(event.get("event_type") == "interrupted_by_persona" for event in record["event_log"])
    assert not any(turn.get("persona_fired") for turn in record["conversation"])


def test_misconception_interrupts_on_its_deadline_tool_only_while_unfired() -> None:
    """Misconception uses deadline_tool_id; a fired persona or another tool never interrupts."""

    package = _package()
    package["persona"] = {"customer_misconception": {
        "utterance": "もう条件は満たしていますよね。", "fires_when": "処理へ進むとき", "statement": "S",
        "trigger": "T", "deadline": "D", "deadline_tool_id": "search",
        "explicit_acknowledgement_required": False, "stop_condition": None,
    }}
    instance = runtime.PackageRuntime(package, scenario_id="c-package")
    kwargs = {"variant_id": "hard", "requested_tool_ids": ["search"]}
    assert runtime._deadline_persona_interrupts(instance, fired_persona=set(), **kwargs) == ["customer_misconception"]
    assert runtime._deadline_persona_interrupts(instance, fired_persona={"customer_misconception"}, **kwargs) == []
    assert runtime._deadline_persona_interrupts(instance, fired_persona=set(), variant_id="hard", requested_tool_ids=["other"]) == []


def test_customer_is_told_about_operator_silence_only_after_silence(monkeypatch, tmp_path) -> None:
    """The silence note appears in the customer instructions only right after an empty operator turn."""

    seen = []
    replies = iter([
        {"choices": [{"message": {"content": ""}}]},
        _CUSTOMER_CONTINUE,
        {"choices": [{"message": {"content": "承知しました。"}}]},
        _CUSTOMER_END,
    ])

    def transport(url, body, headers, timeout):
        if body["messages"][0]["role"] == "system" and "顧客情報" in body["messages"][0]["content"]:
            seen.append("担当者が何も話さなかった" in body["messages"][0]["content"])
        return next(replies)

    runtime.run_package_chat(
        endpoint="http://unused", model="unit", operator_headers=None, scenario_id="c-package",
        scenario_entry={"scenario": _package(), "path": "unit.yaml"}, output_dir=tmp_path,
        timeout_sec=1, extra_body={}, user_controller_endpoint="http://unused",
        user_controller_model="unit", user_controller_extra_body={}, user_controller_headers={},
        max_tool_rounds=2, max_turns=2, variant_id="baseline", run_id="c-silence-note",
        transport=transport,
    )
    assert seen == [True, False]
