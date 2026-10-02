from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml


from elyza_agent_tasks_customer_service.evaluation.audio.omni_audio_common import (
    append_jsonl,
    format_exception,
    resolve_path,
)
from elyza_agent_tasks_customer_service.evaluation.audio.audio_runtime import (
    DEFAULT_CUSTOMER_TTS_MODEL,
    require_dir,
    require_file,
)
from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import (
    DEFAULT_SOP_SEARCH_TOP_K,
    PackageRuntime,
    run_output_dir,
    run_package_chat,
)
from elyza_agent_tasks_customer_service.evaluation.contracts.variants import (
    normalize_variant,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.conversation_log_scoring import (
    load_judge_config,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import score_package_record
from elyza_agent_tasks_customer_service.evaluation.audio.audio_user_channel import AudioUserChannel
from elyza_agent_tasks_customer_service.evaluation.audio.gemini_live_operator_backend import GeminiLiveOperatorBackend
from elyza_agent_tasks_customer_service.evaluation.audio.realtime_operator_backend import RealtimeOperatorBackend
from elyza_agent_tasks_customer_service.evaluation.audio.omni_operator_backend import OmniOperatorBackend
from elyza_agent_tasks_customer_service.evaluation.scoring.metric_bridge import build_measurement_availability
from elyza_agent_tasks_customer_service.evaluation.audio.audio_value_metrics import score_value_metrics
from elyza_agent_tasks_customer_service.evaluation.llm.google_credentials import access_token



DEFAULT_USER_CONTROLLER_MODEL = "gpt-5.6-luna"
DEFAULT_MAX_WORKERS = 6
MAX_WORKERS_LIMIT = 6
DEFAULT_RUN_NUMBER = 1
IO_MODES = ("text-text", "audio-text", "audio-audio")
RECORD_FILE_NAME = "record.json"
SCORE_FILE_NAME = "score.json"
SUMMARY_FILE_NAME = "summary.json"
AUDIO_METRIC_RESULTS_FILE_NAME = "audio_metric_results.json"
AUDIO_METRIC_IDS = ("M20", "M21", "M22", "M23", "M25", "M26")
AUDIO_STATUS_MAP = {
    "passed": "pass",
    "failed": "fail",
    "N/A": "N/A",
    "N/M": "N/M",
    "contract_invalid": "contract_invalid",
}


def load_env_file(path: Path | None) -> None:
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(f"env_path does not exist: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_packages(package_dir: Path, scenario_ids: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Load object-root YAML or JSON packages without adapting their fields.

    ``package_dir`` must contain ``*.yaml``, ``*.yml``, or ``*.json`` mappings
    whose ``scenario_id`` matches the requested ID. Missing files, arrays,
    scalars, duplicate IDs, and mismatches raise ``ValueError`` or
    ``FileNotFoundError``.
    """

    paths = sorted(
        path
        for pattern in ("*.yaml", "*.yml", "*.json")
        for path in package_dir.glob(pattern)
    )
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        if not isinstance(value, dict):
            raise ValueError(f"package must be an object: {path}")
        scenario_id = value.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"scenario_id missing in package: {path}")
        if scenario_id in indexed:
            raise ValueError(f"duplicate scenario_id under {package_dir}: {scenario_id}")
        indexed[scenario_id] = (path, value)
    selected = sorted(indexed) if scenario_ids is None else scenario_ids
    result: dict[str, dict[str, Any]] = {}
    for scenario_id in selected:
        indexed_package = indexed.get(scenario_id)
        if indexed_package is None:
            raise FileNotFoundError(f"{scenario_id}: package file is missing under {package_dir}")
        path, value = indexed_package
        result[scenario_id] = {"scenario": value, "path": path}
    return result


def rebuild_summary(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a summary from object-root ``record.json`` files under ``output_dir``.

    Each record must be a JSON object. Invalid JSON or non-object roots raise
    the normal JSON decoder error or ``ValueError`` rather than omitting a run.
    """

    records: list[dict[str, Any]] = []
    for record_path in sorted(output_dir.rglob(RECORD_FILE_NAME)):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"record.json must be an object: {record_path}")
        records.append(record)
    failures = [record for record in records if record.get("status") != "success"]
    return {
        **summary,
        "scenario_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "scenario_wall_times_sec": {
            str(record.get("scenario_id")): record.get("wall_time_sec")
            for record in records
        },
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    """Atomically replace ``output_dir/summary.json`` with the supplied object."""

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_dir, prefix=".summary-", suffix=".tmp", delete=False
    ) as temporary_file:
        json.dump(summary, temporary_file, ensure_ascii=False, indent=2)
        temporary_path = Path(temporary_file.name)
    try:
        os.replace(temporary_path, output_dir / SUMMARY_FILE_NAME)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _combined_audio_result(
    *,
    signal_result: dict[str, Any],
    package: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Join M21/M23 from ``signal_result`` with M20/M22/M25/M26 scored from the record."""

    signal_metrics = signal_result.get("metrics")
    if not isinstance(signal_metrics, dict) or not all(
        metric_id in signal_metrics for metric_id in ("M21", "M23")
    ):
        raise ValueError("audio signal result must contain M21 and M23")
    value = score_value_metrics(record=record, scenario=package)
    return {
        "schema_version": "audio_metric_results",
        "run_id": value["run_id"],
        "metrics": {
            "M20": value["metrics"]["M20"],
            "M21": signal_metrics["M21"],
            "M22": value["metrics"]["M22"],
            "M23": signal_metrics["M23"],
            "M25": value["metrics"]["M25"],
            "M26": value["metrics"]["M26"],
        },
    }


def _write_package_score(
    *,
    package: dict[str, Any],
    record: dict[str, Any],
    judge_config: dict[str, Any],
    judge_cache_dir: Path,
    mode: str,
    run_dir: Path,
    audio_result: dict[str, Any] | None = None,
) -> Path:
    """Score one object-root package record and replace its ``score.json``.

    ``package``, ``record``, and optional ``audio_result`` are decoded JSON/YAML
    object roots.  ``audio_result`` supplies M21/M23; M20/M22/M25/M26 are scored
    here from ``record`` and ``package``, and the combined audio result replaces
    ``audio_user/audio_metric_results.json``.  Invalid scoring or audio metric
    payloads raise rather than writing a partial score.
    """

    score = score_package_record(
        package=package,
        record=record,
        judge_config=judge_config,
        cache_dir=judge_cache_dir,
        mode=mode,
    )
    if audio_result is not None:
        audio_result = _combined_audio_result(
            signal_result=audio_result, package=package, record=record
        )
        audio_path = run_dir / "audio_user" / AUDIO_METRIC_RESULTS_FILE_NAME
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_text(
            json.dumps(audio_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _merge_audio_results(score, audio_result)
        score["measurement_availability"] = build_measurement_availability(
            score["metric_results"]
        )
        score["not_measured"] = _not_measured_rows(score["metric_results"])
    score_path = run_dir / SCORE_FILE_NAME
    score_path.write_text(
        json.dumps(score, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return score_path


def _run_package_score_only(
    *,
    entries: dict[str, dict[str, Any]],
    output_dir: Path,
    variant_id: str,
    io_mode: str,
    judge_config: dict[str, Any],
    judge_cache_dir: Path,
    model: str,
) -> bool:
    """Rescore selected persisted ``run_001/record.json`` files without execution.

    Each record and optional audio metric file must be JSON object roots.
    Missing records are skipped; malformed records or audio results raise.
    """

    scored_count = 0
    skipped_scenario_ids: list[str] = []
    for scenario_id, entry in entries.items():
        run_dir = run_output_dir(
            output_dir,
            scenario_id=scenario_id,
            variant_id=variant_id,
            run_number=DEFAULT_RUN_NUMBER,
        )
        record_path = run_dir / RECORD_FILE_NAME
        if not record_path.is_file():
            skipped_scenario_ids.append(scenario_id)
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"record.json must be an object: {record_path}")
        audio_result = None
        if io_mode != "text-text":
            audio_path = run_dir / "audio_user" / AUDIO_METRIC_RESULTS_FILE_NAME
            if audio_path.is_file():
                audio_result = json.loads(audio_path.read_text(encoding="utf-8"))
                if not isinstance(audio_result, dict):
                    raise ValueError(
                        f"audio_metric_results.json must be an object: {audio_path}"
                    )
        _write_package_score(
            package=entry["scenario"],
            record=record,
            judge_config=judge_config,
            judge_cache_dir=judge_cache_dir,
            mode=io_mode if io_mode != "text-text" else "text",
            run_dir=run_dir,
            audio_result=audio_result,
        )
        scored_count += 1
    skipped_count = len(skipped_scenario_ids)
    summary = {
        "mode": "chat_tools",
        "score_only": True,
        "io_mode": io_mode,
        "model": model,
        "variant_id": variant_id,
        "run_number": DEFAULT_RUN_NUMBER,
        "scored_count": scored_count,
        "skipped_count": skipped_count,
        "skipped_scenario_ids": skipped_scenario_ids,
    }
    write_summary(output_dir, rebuild_summary(output_dir, summary))
    print(
        json.dumps(
            {
                "score_only": True,
                "scored_count": scored_count,
                "skipped_count": skipped_count,
            }
        )
    )
    return True


def run_package_eval(config: dict[str, Any], endpoint: str = "") -> bool:
    """Run the selected object-root YAML packages directly.

    ``config`` requires ``scenario_dir``, ``output_dir``, ``api_model``, and
    ``variant_id`` (``baseline`` or ``hard``).
    With ``score_only=true``,
    persisted records are rescored and ``api_model`` and runtime endpoints are
    not required.
    ``scenario_ids`` may be an array of package IDs; when omitted every YAML
    package is run. Invalid configuration or packages stop.
    """

    package_dir = require_dir(resolve_path(config.get("scenario_dir")), "scenario_dir")
    output_dir = resolve_path(config.get("output_dir"))
    if output_dir is None:
        raise ValueError("output_dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = resolve_path(config.get("output_jsonl")) or output_dir / "results.jsonl"
    score_only = config.get("score_only", False)
    if not isinstance(score_only, bool):
        raise ValueError("score_only must be a boolean")
    variant_id = normalize_variant(config.get("variant_id"))
    io_mode = str(config.get("io_mode") or "text-text")
    if io_mode not in IO_MODES:
        raise ValueError(f"io_mode must be one of {list(IO_MODES)}")
    operator_transport = str(config.get("operator_transport") or "chat")
    operator_prompt_cache = config.get("operator_prompt_cache", False)
    if not score_only and operator_transport not in {
        "chat",
        "realtime",
        "chat_audio",
        "responses",
        "gemini_live",
    }:
        raise ValueError(f"unknown operator_transport: {operator_transport}")
    if not score_only and not isinstance(operator_prompt_cache, bool):
        raise ValueError("operator_prompt_cache must be a boolean")
    realtime_transport = operator_transport == "realtime"
    gemini_live_transport = operator_transport == "gemini_live"
    chat_audio_transport = operator_transport == "chat_audio"
    if not score_only and io_mode != "text-text" and not (
        realtime_transport or chat_audio_transport or gemini_live_transport
    ):
        raise ValueError(
            "audio io_mode requires operator_transport=realtime, chat_audio, or gemini_live"
        )
    if not score_only and chat_audio_transport and io_mode == "text-text":
        raise ValueError("operator_transport=chat_audio requires an audio io_mode")
    scoring = config.get("conversation_log_scoring")
    judge_config = None
    judge_cache_dir = None
    if scoring is not None:
        if not isinstance(scoring, dict) or set(scoring) != {
            "config_path",
            "cache_dir",
        }:
            raise ValueError(
                "conversation_log_scoring must contain config_path and cache_dir"
            )
        judge_config_path = require_file(
            resolve_path(scoring["config_path"]), "conversation-log judge config"
        )
        judge_cache_dir = resolve_path(scoring["cache_dir"])
        if judge_cache_dir is None:
            raise ValueError("conversation_log_scoring.cache_dir is required")
        judge_cache_dir.mkdir(parents=True, exist_ok=True)
        judge_config = load_judge_config(judge_config_path)
    if score_only and judge_config is None:
        raise ValueError("conversation_log_scoring is required for score_only")
    model = str(config.get("api_model") or os.environ.get("MODEL_NAME") or "")
    if not score_only and not model:
        raise ValueError("api_model or MODEL_NAME is required")
    requested_ids = config.get("scenario_ids")
    if requested_ids is not None and (
        not isinstance(requested_ids, list)
        or not all(isinstance(item, str) and item for item in requested_ids)
    ):
        raise ValueError("scenario_ids must be an array of non-empty strings")
    entries = load_packages(package_dir, requested_ids)
    limit = config.get("limit_scenarios")
    if limit is not None:
        entries = dict(list(entries.items())[: int(limit)])
    if score_only:
        return _run_package_score_only(
            entries=entries,
            output_dir=output_dir,
            variant_id=variant_id,
            io_mode=io_mode,
            judge_config=judge_config,
            judge_cache_dir=judge_cache_dir,
            model=model,
        )
    max_workers = config.get("max_workers", DEFAULT_MAX_WORKERS)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or not 1 <= max_workers <= MAX_WORKERS_LIMIT:
        raise ValueError(f"max_workers must be an integer from 1 to {MAX_WORKERS_LIMIT}")
    for scenario_id, entry in entries.items():
        runtime = PackageRuntime(entry["scenario"], scenario_id=scenario_id)
        runtime.tool_definitions()

    operator_key: str | None = None
    operator_key_env = config.get("operator_api_key_env")
    if operator_key_env is None and "api.openai.com" in endpoint.lower():
        operator_key_env = "OPENAI_API_KEY"
    if operator_key_env is not None:
        operator_key = os.environ.get(str(operator_key_env))
        if not operator_key:
            raise ValueError(f"{operator_key_env} is required for operator endpoint")

    user_endpoint = str(config.get("user_controller_endpoint") or endpoint).rstrip("/")
    user_model = str(config.get("user_controller_model") or DEFAULT_USER_CONTROLLER_MODEL)
    user_headers: dict[str, str] = {}
    user_key_env = config.get("user_controller_api_key_env")
    if user_key_env is None and "api.openai.com" in user_endpoint.lower():
        user_key_env = "OPENAI_API_KEY"
    if user_key_env is not None:
        user_key = os.environ.get(str(user_key_env))
        if not user_key:
            raise ValueError(f"{user_key_env} is required for user controller endpoint")
        user_headers["Authorization"] = f"Bearer {user_key}"

    def run_one(scenario_id: str, entry: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        started = time.perf_counter()
        runtime_completed = False
        partial_record: dict[str, Any] = {}
        run_dir = run_output_dir(
            output_dir,
            scenario_id=scenario_id,
            variant_id=variant_id,
            run_number=DEFAULT_RUN_NUMBER,
        )
        operator_backend = None
        audio_channel = None
        try:
            if realtime_transport or chat_audio_transport or gemini_live_transport:
                package_runtime = PackageRuntime(entry["scenario"], scenario_id=scenario_id)
                if (realtime_transport or gemini_live_transport) and not operator_key:
                    raise ValueError("operator API key is required for realtime transport")
                backend_arguments = {
                    "endpoint": endpoint,
                    "model": model,
                    "api_key": operator_key or "EMPTY",
                    "instructions": package_runtime.operator_system_prompt(),
                    "tools": package_runtime.tool_definitions(),
                    "out_dir": run_dir,
                    "temperature": float(config.get("temperature", 0.0)),
                    "output_modality": "audio" if io_mode == "audio-audio" else "text",
                }
                if gemini_live_transport:
                    operator_backend = GeminiLiveOperatorBackend(
                        **backend_arguments,
                        voice=str(config.get("voice") or "Puck"),
                        post_call_model=str(
                            config.get("post_call_model") or "google/gemini-3.6-flash"
                        ),
                    )
                elif realtime_transport:
                    operator_backend = RealtimeOperatorBackend(
                        **backend_arguments,
                        voice=str(config.get("voice") or "alloy"),
                    )
                else:
                    operator_backend = OmniOperatorBackend(
                        **backend_arguments,
                        seed=int(config.get("seed", 0)),
                        max_tokens=int(config.get("max_tokens", 4096)),
                        provider_profile=str(config.get("provider_profile") or "vllm"),
                    )
                if io_mode != "text-text":
                    audio_channel = AudioUserChannel(
                        out_dir=run_dir,
                        tts_cache_dir=output_dir / ".tts_cache",
                        tts_model=str(
                            config.get("customer_tts_model")
                            or DEFAULT_CUSTOMER_TTS_MODEL
                        ),
                    )
            operator_headers = {}
            if operator_key:
                operator_token = access_token() if operator_key == "ADC" else operator_key
                operator_headers["Authorization"] = f"Bearer {operator_token}"
            scenario_records = run_package_chat(
                endpoint=endpoint,
                model=model,
                operator_headers=operator_headers,
                operator_uses_responses=(operator_transport == "responses"),
                operator_reasoning=config.get("operator_reasoning"),
                operator_prompt_cache=operator_prompt_cache,
                scenario_id=scenario_id,
                scenario_entry=entry,
                output_dir=output_dir,
                timeout_sec=float(config.get("timeout_sec", 600)),
                extra_body=dict(config.get("request_extra_body", {})),
                user_controller_endpoint=user_endpoint,
                user_controller_model=user_model,
                user_controller_extra_body=dict(config.get("user_controller_request_extra_body", {})),
                user_controller_headers=user_headers,
                max_tool_rounds=int(config.get("max_tool_rounds", 40)),
                max_turns=int(config.get("max_turns", 28)),
                variant_id=variant_id,
                run_number=DEFAULT_RUN_NUMBER,
                sop_search_top_k=config.get("sop_search_top_k", DEFAULT_SOP_SEARCH_TOP_K),
                partial_record=partial_record,
                operator_backend=operator_backend,
                audio_user_channel=audio_channel,
                io_mode=io_mode,
            )
            runtime_completed = True
            if judge_config is not None:
                for record in scenario_records:
                    audio_result = None
                    score_mode = "text"
                    if audio_channel is not None:
                        score_mode = io_mode
                        audio_result = audio_channel.finalize(
                            run_id=record["run_id"],
                            scenario_id=scenario_id,
                            source_rows=record["event_log"],
                        )
                    score_path = _write_package_score(
                        package=entry["scenario"],
                        record=record,
                        judge_config=judge_config,
                        judge_cache_dir=judge_cache_dir,
                        mode=score_mode,
                        run_dir=run_dir,
                        audio_result=audio_result,
                    )
                    record["conversation_log_score_path"] = str(score_path)
            return scenario_records, None
        except Exception as exc:
            if runtime_completed:
                raise
            error_type, error_message = format_exception(exc)
            failure = {
                **partial_record,
                "run_mode": "chat_tools",
                "model": model,
                "scenario_id": scenario_id,
                "variant_id": variant_id,
                "run_number": DEFAULT_RUN_NUMBER,
                "status": "failed_runtime",
                "error_type": error_type,
                "error_message": error_message,
                "failure_category": classify_runtime_failure(error_type=error_type, error_message=error_message),
                "wall_time_sec": round(time.perf_counter() - started, 6),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            if judge_config is not None:
                score_path = _write_package_score(
                    package=entry["scenario"],
                    record=failure,
                    judge_config=judge_config,
                    judge_cache_dir=judge_cache_dir,
                    mode=io_mode if io_mode != "text-text" else "text",
                    run_dir=run_dir,
                )
                failure["conversation_log_score_path"] = str(score_path)
            (run_dir / RECORD_FILE_NAME).write_text(
                json.dumps(failure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return [], failure
        finally:
            if operator_backend is not None:
                operator_backend.close()

    run_started = time.perf_counter()
    outcomes: dict[str, tuple[list[dict[str, Any]], dict[str, Any] | None]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one, scenario_id, entry): scenario_id
            for scenario_id, entry in entries.items()
        }
        for future in as_completed(futures):
            outcomes[futures[future]] = future.result()

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for scenario_id in entries:
        scenario_records, failure = outcomes[scenario_id]
        records.extend(scenario_records)
        if failure is not None:
            failures.append(failure)
            append_jsonl(output_jsonl, failure)
            continue
        for row in scenario_records:
            append_jsonl(output_jsonl, {"run_mode": "chat_tools", "model": model, **row})
    summary = {
        "mode": "chat_tools",
        "io_mode": io_mode,
        "model": model,
        "variant_id": variant_id,
        "run_number": DEFAULT_RUN_NUMBER,
        "scenario_count": len(entries),
        "success_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "max_workers": max_workers,
        "wall_time_sec": round(time.perf_counter() - run_started, 6),
        "scenario_wall_times_sec": {
            scenario_id: (outcomes[scenario_id][1] or outcomes[scenario_id][0][0])["wall_time_sec"]
            for scenario_id in entries
        },
        "output_jsonl": str(output_jsonl),
    }
    write_summary(output_dir, rebuild_summary(output_dir, summary))
    return not failures


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("Usage: run_eval.py '<eval_config_json>' <endpoint>")
    config = json.loads(sys.argv[1])
    if len(sys.argv) == 2 and not config.get("score_only"):
        raise SystemExit("Usage: run_eval.py '<eval_config_json>' <endpoint>")
    endpoint = sys.argv[2].rstrip("/") if len(sys.argv) == 3 else ""
    load_env_file(resolve_path(config.get("env_path")))
    if config.get("mode") != "chat_tools":
        raise SystemExit('mode must be "chat_tools"')
    return 0 if run_package_eval(config, endpoint) else 1


def _merge_audio_results(
    report: dict[str, Any],
    audio_result: dict[str, Any],
) -> None:
    source_rows = audio_result.get("metrics")
    if not isinstance(source_rows, dict):
        raise ValueError("audio scorer result.metrics must be an object")
    violations = [
        row
        for row in report["violations"]
        if row.get("metric_id") not in set(AUDIO_METRIC_IDS)
    ]
    for row in report["metric_results"]:
        metric_id = row["metric_id"]
        if metric_id not in AUDIO_METRIC_IDS:
            continue
        source = source_rows.get(metric_id)
        if not isinstance(source, dict):
            raise ValueError(f"audio scorer result missing {metric_id}")
        source_status = source.get("status")
        if source_status not in AUDIO_STATUS_MAP:
            raise ValueError(f"audio scorer status is invalid for {metric_id}: {source_status}")
        published_status = AUDIO_STATUS_MAP[source_status]
        published_reason = str(source.get("reason") or source_status)
        row.update(
            {
                "status": published_status,
                "value": deepcopy(source),
                "reason": published_reason,
                "violations": [],
                "diagnostics": {
                    "source_scorer": audio_result.get("schema_version"),
                    "audio_run_id": audio_result.get("run_id"),
                },
            }
        )
        if source_status in {"failed", "contract_invalid"}:
            violation_type = "audio_metric_failed"
            if source_status == "contract_invalid":
                violation_type = "audio_metric_contract_invalid"
            violation = {
                "metric_instance_id": row["metric_instance_id"],
                "metric_id": metric_id,
                "violation_type": violation_type,
                "evidence": {"reason": row["reason"]},
            }
            row["violations"].append(deepcopy(violation))
            violations.append(violation)
    report["violations"] = violations
    report["diagnostics"]["audio_metric_schema_version"] = audio_result.get(
        "schema_version"
    )


def _not_measured_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "metric_instance_id": row["metric_instance_id"],
            "metric_id": row["metric_id"],
            "reason": row["reason"],
        }
        for row in rows
        if row["status"] in {"N/M", "contract_invalid"}
    ]


def classify_runtime_failure(*, error_type: str, error_message: str) -> str:
    text = f"{error_type}\n{error_message}"
    if any(token in text for token in ("ConnectionClosedError", "BrokenPipeError", "WebSocket", "no close frame")):
        return "transport_failure"
    if any(token in text for token in ("ASR", "asr", "transcribe", "transcript is missing")):
        return "asr_failure"
    if any(token in text for token in ("TTS", "tts", "synthesize")):
        return "tts_failure"
    if any(token in text for token in ("JSON", "json", "schema", "could not be repaired")):
        return "judge_or_controller_json_failure"
    if any(token in text for token in ("TimeoutError", "timed out", "timeout")):
        return "timeout_failure"
    return "runtime_failure"


if __name__ == "__main__":
    raise SystemExit(main())
