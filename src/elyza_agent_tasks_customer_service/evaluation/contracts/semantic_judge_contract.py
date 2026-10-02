"""Validate LLM judge member configuration and derive its sampling parameters."""

from __future__ import annotations

from typing import Any


OPENAI_MODELS_WITHOUT_TEMPERATURE = ("gpt-5",)


def validate_judge_member(value: Any, *, label: str = "semantic judge member") -> dict[str, Any]:
    """Validate one explicitly configured fixed-revision observer member."""

    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    required = {
        "member_id",
        "family",
        "provider",
        "endpoint",
        "transport",
        "resolved_model_revision",
        "revision_policy",
        "api_key_env",
        "sampling",
    }
    _require_exact_keys(value, required, label)
    for key in ("member_id", "family", "provider", "endpoint", "resolved_model_revision", "api_key_env"):
        _require_string(value[key], f"{label}.{key}")
    if value["transport"] != "chat_completions":
        raise ValueError(f"{label}.transport must be chat_completions")
    if value["revision_policy"] != "fixed":
        raise ValueError(f"{label} must declare revision_policy=fixed")
    _validate_sampling(value["sampling"], f"{label}.sampling")
    if (
        value["provider"] == "openai"
        and value["resolved_model_revision"].startswith(OPENAI_MODELS_WITHOUT_TEMPERATURE)
        and value["sampling"]["temperature"] is not None
    ):
        raise ValueError(
            f"{label} model {value['resolved_model_revision']} does not accept temperature; "
            f"set {label}.sampling.temperature to null to explicitly omit it"
        )
    return value


def judge_sampling_parameters(member: dict[str, Any]) -> dict[str, int | float]:
    """Return configured chat sampling fields, omitting only explicit null temperature."""

    validate_judge_member(member)
    sampling = member["sampling"]
    result: dict[str, int | float] = {"seed": sampling["seed"]}
    if sampling["temperature"] is not None:
        result["temperature"] = sampling["temperature"]
    return result


def _validate_sampling(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(value, {"temperature", "seed", "max_tokens"}, label)
    temperature = value["temperature"]
    if temperature is not None and (
        not isinstance(temperature, int | float) or isinstance(temperature, bool)
    ):
        raise ValueError(f"{label}.temperature must be numeric or null")
    seed = value["seed"]
    max_tokens = value["max_tokens"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{label}.seed must be an integer")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError(f"{label}.max_tokens must be a positive integer")


def _require_exact_keys(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise ValueError(f"{label} fields invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")


def _require_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
