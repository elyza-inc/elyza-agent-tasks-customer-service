"""High-precision M17 rules over pinned GiNZA parses and explicit relations."""

from __future__ import annotations

from elyza_agent_tasks_customer_service.evaluation.config_paths import config_path
import re
from pathlib import Path
from typing import Any

import yaml

from elyza_agent_tasks_customer_service.evaluation.contracts.artifact_receipts import sha256_json
from elyza_agent_tasks_customer_service.evaluation.observers.japanese_parse_adapter import (
    PARSE_SCHEMA_VERSION,
)


DEFAULT_RULE_CATALOG_PATH = (
    config_path("japanese_register_rules.yaml")
)
REGISTER_CONTEXT_VERSION = "japanese_register_context"
RULE_CATALOG_VERSION = "japanese_register_rules"
OBSERVER_VERSION = "japanese_rule_observer"
OBSERVER_RECEIPT_VERSION = "japanese_rule_observer_receipt"
UTTERANCE_POLICY_VERSION = "japanese_utterance_quality_policy"
RULE_FAMILIES = ("closed_misuse", "honorific_direction", "in_out_group")
RELATIONS = {"self", "in_group", "customer", "out_group"}
JAPANESE_TEXT_PATTERN = re.compile(r"[ぁ-んァ-ヶ一-龯々〆ヵヶ]")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"[。！？]")
POLITE_END_PATTERN = re.compile(
    r"(?:です|でした|ます|ました|ません|ませんでした|ください|くださいませ|でしょう)"
    r"(?:よね|かね|か|ね|よ)?[」』）)\]］】]*$"
)
TRAILING_EXAMPLE_PATTERN = re.compile(r"(?:（|\()(?:例[:：]).*(?:）|\))$")
EXAMPLE_PREFIXES = ("例:", "例：")
BRACKET_PAIRS = {
    "「": "」",
    "『": "』",
    "（": "）",
    "(": ")",
    "［": "］",
    "[": "]",
    "【": "】",
}
DUPLICATE_CASE_PARTICLES = frozenset(("を", "が", "は", "に", "で", "と", "へ"))
MIN_DUPLICATE_SENTENCE_LENGTH = 20
MALFORMED_JAPANESE_PATTERNS = (
    (
        "M17-JA-BROKEN-POLITE-SEQUENCE-002",
        re.compile(r"(?:ですます|ますです|ございますです|くださいです|させていただかせ)"),
        "malformed_polite_sequence",
    ),
)
SPOKEN_STYLE_PATTERNS = (
    (
        "M24-SPOKEN-STYLE-CODE-BLOCK-001",
        re.compile(r"^[ \t　]*```[^\r\n]*", re.MULTILINE),
    ),
    (
        "M24-SPOKEN-STYLE-TABLE-002",
        re.compile(r"^[^\r\n]*[|｜][^\r\n]*$", re.MULTILINE),
    ),
    (
        "M24-SPOKEN-STYLE-HEADING-003",
        re.compile(r"^[ \t　]*[#＃]+[ \t　]*\S[^\r\n]*", re.MULTILINE),
    ),
    (
        "M24-SPOKEN-STYLE-BOLD-004",
        re.compile(r"\*\*[^\r\n*]+\*\*"),
    ),
    (
        "M24-SPOKEN-STYLE-NUMBERED-005",
        re.compile(
            r"^[ \t　]*(?:[0-9０-９]+[.．][ \t　]+|"
            r"[0-9０-９]+[)）][ \t　]*|[(（][0-9０-９]+[)）][ \t　]*)"
            r"[ \t　]*\S[^\r\n]*",
            re.MULTILINE,
        ),
    ),
    (
        "M24-SPOKEN-STYLE-BULLET-006",
        re.compile(
            r"^[ \t　]*(?:(?:[-*－＊﹣−][ \t　]+)|・[ \t　]*)\S[^\r\n]*",
            re.MULTILINE,
        ),
    ),
    (
        "M24-SPOKEN-STYLE-URL-007",
        re.compile(r"(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE),
    ),
    (
        "M24-SPOKEN-STYLE-PARAGRAPH-008",
        re.compile(r"\r?\n[ \t　]*\r?\n"),
    ),
    (
        "M24-SPOKEN-STYLE-HARD-BREAK-009",
        re.compile(r"[ \t]{2,}\r?\n(?=[ \t　]*\S)"),
    ),
    (
        "M24-SPOKEN-STYLE-NUM-011",
        re.compile(r"(?:\d(?:[-:/,.]\d)+|\d[-:/,.]|[-:/,.]\d)"),
    ),
    (
        "M24-SPOKEN-STYLE-IDENT-012",
        re.compile(r"(?<![a-z_])(?=[a-z_]{6,}(?![a-z_]))[a-z_]*_[a-z_]+(?![a-z_])"),
    ),
    (
        "M24-SPOKEN-STYLE-LONGSENT-013",
        re.compile(r"[^。！？]{201,}"),
    ),
)
POLITE_REGISTER_PATTERNS = (
    (
        "M17-REGISTER-DOUBLE-HONORIFIC-002",
        re.compile(
            r"ご(?:予約|依頼|搭乗)"
            r"され(?:ている|た|る)(?:方|ご本人)"
        ),
        "double_honorific_go_noun_plus_rareru",
    ),
)
UTTERANCE_CRITERIA = {
    "japanese_correctness": (
        "contains_japanese_text",
        "pinned_parser_produced_non_whitespace_tokens",
        "paired_brackets_are_balanced",
        "no_adjacent_duplicate_case_particle",
        "no_malformed_polite_sequence",
    ),
    "polite_register": (
        "every_non_example_customer_facing_sentence_uses_polite_ending",
        "no_closed_double_honorific_go_noun_plus_rareru",
    ),
}


def validate_register_context(value: Any) -> dict[str, Any]:
    """Validate explicit people relations used by M17.

    The accepted object declares operator-to-customer direction, explicit
    speaker/addressee IDs, two or more people with unique IDs and non-empty
    surface forms, and one or more adopted required rule families. Unknown
    fields, undeclared endpoint IDs, duplicate surfaces within one person, and
    unsupported relations/families raise ``ValueError``.
    """

    if not isinstance(value, dict):
        raise ValueError("Japanese register context must be an object")
    required = {
        "schema_version",
        "communication_direction",
        "speaker_person_id",
        "addressee_person_id",
        "persons",
        "required_rule_families",
    }
    if set(value) != required:
        raise ValueError("Japanese register context fields are invalid")
    if value["schema_version"] != REGISTER_CONTEXT_VERSION:
        raise ValueError("unsupported Japanese register context")
    if value["communication_direction"] != "operator_to_customer":
        raise ValueError("M17 supports only explicit operator_to_customer context")
    persons = value["persons"]
    if not isinstance(persons, list) or len(persons) < 2:
        raise ValueError("Japanese register context requires at least two persons")
    person_ids: list[str] = []
    relation_by_id: dict[str, str] = {}
    for index, person in enumerate(persons):
        if not isinstance(person, dict) or set(person) != {
            "person_id",
            "relation_to_operator",
            "surface_forms",
        }:
            raise ValueError(f"Japanese register persons[{index}] fields are invalid")
        person_id = person["person_id"]
        if not isinstance(person_id, str) or not person_id:
            raise ValueError(f"Japanese register persons[{index}].person_id is invalid")
        relation = person["relation_to_operator"]
        if relation not in RELATIONS:
            raise ValueError(f"Japanese register persons[{index}] relation is invalid")
        surfaces = person["surface_forms"]
        if not isinstance(surfaces, list) or not surfaces:
            raise ValueError(f"Japanese register persons[{index}] surfaces are invalid")
        if not all(isinstance(surface, str) and surface for surface in surfaces):
            raise ValueError(f"Japanese register persons[{index}] surfaces are invalid")
        if len(surfaces) != len(set(surfaces)):
            raise ValueError(f"Japanese register persons[{index}] surfaces are duplicated")
        person_ids.append(person_id)
        relation_by_id[person_id] = relation
    if len(person_ids) != len(set(person_ids)):
        raise ValueError("Japanese register person IDs must be unique")
    speaker_id = value["speaker_person_id"]
    addressee_id = value["addressee_person_id"]
    if speaker_id not in relation_by_id or relation_by_id[speaker_id] != "self":
        raise ValueError("Japanese register speaker must reference an explicit self person")
    if addressee_id not in relation_by_id or relation_by_id[addressee_id] != "customer":
        raise ValueError("Japanese register addressee must reference an explicit customer")
    families = value["required_rule_families"]
    if not isinstance(families, list) or not families:
        raise ValueError("Japanese register required_rule_families must be non-empty")
    if len(families) != len(set(families)) or set(families) - set(RULE_FAMILIES):
        raise ValueError("Japanese register required_rule_families are invalid")
    return value


def load_rule_catalog(
    path: Path = DEFAULT_RULE_CATALOG_PATH,
) -> dict[str, Any]:
    """Load the M17 rule catalog from one object-root ``.yaml`` file.

    JSON/scalar/array roots, malformed YAML, unknown top-level fields,
    missing adopted families, malformed source-backed rules, and invalid
    regular expressions raise ``ValueError``.
    """

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Japanese rule catalog must be YAML: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load Japanese rule catalog {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Japanese rule catalog must be an object")
    required = {
        "schema_version",
        "catalog_version",
        "normative_sources",
        "rule_families",
    }
    if set(value) != required:
        raise ValueError("Japanese rule catalog fields are invalid")
    if value["schema_version"] != RULE_CATALOG_VERSION:
        raise ValueError("unsupported Japanese rule catalog")
    sources = value["normative_sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("Japanese rule catalog requires normative sources")
    source_refs = {
        source.get("source_ref")
        for source in sources
        if isinstance(source, dict)
    }
    if None in source_refs:
        raise ValueError("Japanese rule source_ref is missing")
    families = value["rule_families"]
    if not isinstance(families, dict) or set(families) != set(RULE_FAMILIES):
        raise ValueError("Japanese rule catalog family set is invalid")
    closed = families["closed_misuse"]
    if not isinstance(closed, dict) or set(closed) != {
        "tracked_patterns",
        "violation_rules",
    }:
        raise ValueError("closed_misuse catalog fields are invalid")
    for pattern in closed["tracked_patterns"]:
        re.compile(pattern)
    rule_ids: list[str] = []
    for rule in closed["violation_rules"]:
        if not isinstance(rule, dict) or set(rule) != {
            "rule_id",
            "pattern",
            "reason_code",
            "source_ref",
        }:
            raise ValueError("closed misuse rule fields are invalid")
        if rule["source_ref"] not in source_refs:
            raise ValueError("closed misuse rule source is undeclared")
        re.compile(rule["pattern"])
        rule_ids.append(rule["rule_id"])
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Japanese rule IDs must be unique")
    direction = families["honorific_direction"]
    expected_direction = {
        "honorific_lemmas",
        "humble_lemmas",
        "honorific_allowed_relations",
        "humble_allowed_relations",
        "source_ref",
    }
    if not isinstance(direction, dict) or set(direction) != expected_direction:
        raise ValueError("honorific_direction catalog fields are invalid")
    if direction["source_ref"] not in source_refs:
        raise ValueError("honorific_direction source is undeclared")
    in_out = families["in_out_group"]
    if not isinstance(in_out, dict) or set(in_out) != {
        "honorific_suffixes",
        "forbidden_relations",
        "source_ref",
    }:
        raise ValueError("in_out_group catalog fields are invalid")
    if in_out["source_ref"] not in source_refs:
        raise ValueError("in_out_group source is undeclared")
    return value


def score_japanese_register_from_parse(
    *,
    parse_bundle: dict[str, Any],
    context: dict[str, Any],
    rule_catalog: dict[str, Any],
) -> dict[str, Any]:
    """Score every turn in one fixed parse; invalid bundles raise.

    ``parse_bundle`` is the exact ``parse``/``receipt`` object emitted by
    ``parse_japanese_turns``. ``context`` and ``rule_catalog`` use the exact
    object shapes validated by this module. Unknown fields, malformed values,
    or receipt/hash inconsistencies raise ``ValueError``.
    """

    validate_register_context(context)
    _validate_rule_catalog_value(rule_catalog)
    parsed, parse_receipt = _validate_parse_bundle(parse_bundle)
    family_rows = [
        _closed_misuse_family(parsed, rule_catalog),
        _honorific_direction_family(parsed, context, rule_catalog),
        _in_out_group_family(parsed, context, rule_catalog),
    ]
    unresolved = {
        row["rule_family"]: _family_unresolved(row)
        for row in family_rows
        if _family_unresolved(row)
    }
    judge_candidates = [
        candidate
        for row in family_rows
        for candidate in row["diagnostics"].get("judge_candidates", [])
    ]
    utterance_rows = _utterance_assessments(parsed)
    required = context["required_rule_families"]
    family_violation_count = sum(
        row["violation_count"] for row in family_rows
    )
    utterance_violation_count = sum(
        row["violation_count"] for row in utterance_rows
    )
    utterance_pass_count = sum(row["passed"] for row in utterance_rows)
    utterance_pass_ratio = None
    if utterance_rows:
        utterance_pass_ratio = utterance_pass_count / len(utterance_rows)
    # Relation-aware family violations remain independent evidence. They can
    # keep the scenario status at fail, but never alter the utterance ratio.
    confirmed_violations = family_violation_count + utterance_violation_count
    if confirmed_violations:
        status = "fail"
        passed: bool | None = False
        reason = "confirmed_japanese_or_register_violation"
    elif utterance_rows:
        status = "pass"
        passed = True
        reason = "all_operator_utterances_evaluated_and_clear"
    else:
        status = "N/M"
        passed = None
        reason = "operator_utterances_unavailable"
    result_body = {
        "schema_version": OBSERVER_VERSION,
        "metric_id": "M17",
        "status": status,
        "passed": passed,
        "reason": reason,
        "scope": "all_operator_utterances_japanese_correctness_and_polite_register",
        "utterance_policy_version": UTTERANCE_POLICY_VERSION,
        "utterance_criteria": UTTERANCE_CRITERIA,
        "utterance_assessments": utterance_rows,
        "utterance_count": len(utterance_rows),
        "utterance_pass_count": utterance_pass_count,
        "utterance_pass_ratio": utterance_pass_ratio,
        "required_rule_families": required,
        "rule_families": family_rows,
        "unresolved": unresolved,
        "judge_candidates": judge_candidates,
        "family_violation_count": family_violation_count,
        "utterance_violation_count": utterance_violation_count,
        "violation_count": confirmed_violations,
    }
    receipt_body = {
        "schema_version": OBSERVER_RECEIPT_VERSION,
        "observer_version": OBSERVER_VERSION,
        "parse_receipt_id": parse_receipt["receipt_id"],
        "parser_lock_hash": parse_receipt["parser_lock_hash"],
        "parser_resource_hashes": parse_receipt["resource_hashes"],
        "parser_distribution_versions": parse_receipt["distribution_versions"],
        "rule_catalog_version": rule_catalog["catalog_version"],
        "rule_catalog_hash": sha256_json(rule_catalog),
        "context_hash": sha256_json(context),
        "parse_hash": parse_receipt["parse_hash"],
        "result_hash": sha256_json(result_body),
    }
    return {
        **result_body,
        "receipt": {
            **receipt_body,
            "receipt_id": sha256_json(receipt_body),
        },
    }


def _utterance_assessments(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in parsed["turns"]:
        text = turn["text"]
        correctness_violations = _japanese_correctness_violations(turn)
        spoken_style_violations = _spoken_style_violations(turn)
        register_violations = _polite_register_violations(
            turn,
            excluded_spans=[
                row
                for row in spoken_style_violations
                if row["reason_code"] == "written_style_markup_in_spoken_reply"
            ],
        )
        violation_count = len(correctness_violations) + len(register_violations)
        rows.append(
            {
                "turn_ref": turn["turn_ref"],
                "text": text,
                "japanese_correctness": {
                    "passed": not correctness_violations,
                    "violations": correctness_violations,
                },
                "polite_register": {
                    "passed": not register_violations,
                    "violations": register_violations,
                },
                "spoken_style": {
                    "passed": not spoken_style_violations,
                    "violations": spoken_style_violations,
                },
                "violation_count": violation_count,
                "passed": violation_count == 0,
            }
        )
    return rows


def _japanese_correctness_violations(turn: dict[str, Any]) -> list[dict[str, Any]]:
    text = turn["text"]
    violations: list[dict[str, Any]] = []
    if JAPANESE_TEXT_PATTERN.search(text) is None:
        violations.append(
            _utterance_violation(
                rule_id="M17-JA-TEXT-001",
                dimension="japanese_correctness",
                reason_code="japanese_text_absent",
                turn_ref=turn["turn_ref"],
                start=0,
                end=len(text),
                text=text,
            )
        )
    meaningful_tokens = [
        token
        for sentence in turn["sentences"]
        if sentence["text"].strip()
        for token in sentence["tokens"]
        if token["text"].strip()
    ]
    if not meaningful_tokens:
        violations.append(
            _utterance_violation(
                rule_id="M17-JA-PARSE-002",
                dimension="japanese_correctness",
                reason_code="parsed_japanese_structure_absent",
                turn_ref=turn["turn_ref"],
                start=0,
                end=len(text),
                text=text,
            )
        )
    bracket_error = _first_bracket_error(text)
    if bracket_error is not None:
        violations.append(
            _utterance_violation(
                rule_id="M17-JA-BRACKET-003",
                dimension="japanese_correctness",
                reason_code="unbalanced_bracket",
                turn_ref=turn["turn_ref"],
                start=bracket_error,
                end=min(bracket_error + 1, len(text)),
                text=text[bracket_error : bracket_error + 1],
            )
        )
    violations.extend(_duplicate_case_particle_violations(turn))
    for rule_id, pattern, reason_code in MALFORMED_JAPANESE_PATTERNS:
        for match in pattern.finditer(text):
            violations.append(
                _utterance_violation(
                    rule_id=rule_id,
                    dimension="japanese_correctness",
                    reason_code=reason_code,
                    turn_ref=turn["turn_ref"],
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                )
            )
    return violations


def _duplicate_case_particle_violations(turn: dict[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for sentence in turn["sentences"]:
        tokens = sentence["tokens"]
        for previous, current in zip(tokens, tokens[1:]):
            if previous["pos"] != "ADP" or current["pos"] != "ADP":
                continue
            if previous["lemma"] not in DUPLICATE_CASE_PARTICLES:
                continue
            if current["lemma"] != previous["lemma"]:
                continue
            if previous["end_char"] != current["start_char"]:
                continue
            violations.append(
                _utterance_violation(
                    rule_id="M17-JA-REPEATED-PARTICLE-001",
                    dimension="japanese_correctness",
                    reason_code="adjacent_duplicate_case_particle",
                    turn_ref=turn["turn_ref"],
                    start=previous["start_char"],
                    end=current["end_char"],
                    text=previous["text"] + current["text"],
                )
            )
    return violations


def _spoken_style_violations(turn: dict[str, Any]) -> list[dict[str, Any]]:
    text = turn["text"]
    violations: list[dict[str, Any]] = []
    for rule_id, pattern in SPOKEN_STYLE_PATTERNS:
        for match in pattern.finditer(text):
            if any(
                match.start() < row["end_char"] and match.end() > row["start_char"]
                for row in violations
            ):
                continue
            violations.append(
                _utterance_violation(
                    rule_id=rule_id,
                    dimension="spoken_style",
                    reason_code="written_style_markup_in_spoken_reply",
                    turn_ref=turn["turn_ref"],
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                )
            )
    violations.extend(_duplicate_sentence_violations(turn))
    return sorted(violations, key=lambda row: (row["start_char"], row["end_char"]))


def _duplicate_sentence_violations(turn: dict[str, Any]) -> list[dict[str, Any]]:
    text = turn["text"]
    seen: set[str] = set()
    violations: list[dict[str, Any]] = []
    start = 0
    for boundary in SENTENCE_BOUNDARY_PATTERN.finditer(text):
        sentence = text[start : boundary.start()]
        stripped = sentence.strip()
        content_start = start + len(sentence) - len(sentence.lstrip())
        normalized = re.sub(r"\s+", "", stripped)
        if len(normalized) >= MIN_DUPLICATE_SENTENCE_LENGTH:
            if normalized in seen:
                end = boundary.end()
                violations.append(
                    _utterance_violation(
                        rule_id="M24-SPOKEN-STYLE-DUPLICATE-SENTENCE-010",
                        dimension="spoken_style",
                        reason_code="duplicate_sentence_in_spoken_reply",
                        turn_ref=turn["turn_ref"],
                        start=content_start,
                        end=end,
                        text=text[content_start:end],
                    )
                )
            seen.add(normalized)
        start = boundary.end()
    return violations


def _polite_register_violations(
    turn: dict[str, Any],
    *,
    excluded_spans: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    excluded_spans = excluded_spans or []
    for rule_id, pattern, reason_code in POLITE_REGISTER_PATTERNS:
        for match in pattern.finditer(turn["text"]):
            violations.append(
                _utterance_violation(
                    rule_id=rule_id,
                    dimension="polite_register",
                    reason_code=reason_code,
                    turn_ref=turn["turn_ref"],
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                )
            )
    for start, end, sentence in _customer_facing_sentences(turn["text"]):
        if any(
            start < row["end_char"] and end > row["start_char"]
            for row in excluded_spans
        ):
            continue
        candidate = TRAILING_EXAMPLE_PATTERN.sub("", sentence).rstrip()
        if not candidate or candidate.startswith(EXAMPLE_PREFIXES):
            continue
        if JAPANESE_TEXT_PATTERN.search(candidate) is None:
            continue
        if POLITE_END_PATTERN.search(candidate) is not None:
            continue
        violations.append(
            _utterance_violation(
                rule_id="M17-REGISTER-POLITE-END-001",
                dimension="polite_register",
                reason_code="customer_facing_sentence_not_in_polite_register",
                turn_ref=turn["turn_ref"],
                start=start,
                end=end,
                text=sentence,
            )
        )
    return violations


def _customer_facing_sentences(text: str) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    start = 0
    for boundary in SENTENCE_BOUNDARY_PATTERN.finditer(text):
        end = boundary.start()
        sentence = text[start:end].strip()
        if sentence:
            content_start = start + len(text[start:end]) - len(text[start:end].lstrip())
            rows.append((content_start, end, sentence))
        start = boundary.end()
    trailing = text[start:].strip()
    if trailing:
        content_start = start + len(text[start:]) - len(text[start:].lstrip())
        rows.append((content_start, len(text), trailing))
    return rows


def _first_bracket_error(text: str) -> int | None:
    closing_to_opening = {closing: opening for opening, closing in BRACKET_PAIRS.items()}
    stack: list[tuple[str, int]] = []
    for index, character in enumerate(text):
        if character in BRACKET_PAIRS:
            stack.append((character, index))
            continue
        opening = closing_to_opening.get(character)
        if opening is None:
            continue
        if not stack or stack[-1][0] != opening:
            return index
        stack.pop()
    if stack:
        return stack[-1][1]
    return None


def _utterance_violation(
    *,
    rule_id: str,
    dimension: str,
    reason_code: str,
    turn_ref: str,
    start: int,
    end: int,
    text: str,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "dimension": dimension,
        "reason_code": reason_code,
        **_span(turn_ref, start, end, text),
    }


def _closed_misuse_family(
    parsed: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    config = catalog["rule_families"]["closed_misuse"]
    applicable_spans: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for turn in parsed["turns"]:
        text = turn["text"]
        for pattern in config["tracked_patterns"]:
            for match in re.finditer(pattern, text):
                applicable_spans.append(
                    _span(turn["turn_ref"], match.start(), match.end(), match.group(0))
                )
        for rule in config["violation_rules"]:
            for match in re.finditer(rule["pattern"], text):
                violations.append(
                    {
                        "rule_id": rule["rule_id"],
                        "source_ref": rule["source_ref"],
                        "reason_code": rule["reason_code"],
                        **_span(
                            turn["turn_ref"],
                            match.start(),
                            match.end(),
                            match.group(0),
                        ),
                    }
                )
    applicable = bool(applicable_spans)
    passed = not violations if applicable else None
    return _family_row(
        family="closed_misuse",
        applicable=applicable,
        measurable=True,
        passed=passed,
        violations=violations,
        diagnostics={"applicable_spans": applicable_spans},
    )


def _honorific_direction_family(
    parsed: dict[str, Any],
    context: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    config = catalog["rule_families"]["honorific_direction"]
    honorific = set(config["honorific_lemmas"])
    humble = set(config["humble_lemmas"])
    tracked: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    judge_candidates: list[dict[str, str]] = []
    violations: list[dict[str, Any]] = []
    role_default_attribution_count = 0
    persons = _person_surfaces(context)
    for turn in parsed["turns"]:
        for sentence in turn["sentences"]:
            for token in sentence["tokens"]:
                lemma = token["lemma"]
                if lemma not in honorific and lemma not in humble:
                    continue
                speech_level = "honorific" if lemma in honorific else "humble"
                tracked_row = {
                    "turn_ref": turn["turn_ref"],
                    "sentence_id": sentence["sentence_id"],
                    "predicate_token_id": token["token_id"],
                    "predicate": token["text"],
                    "lemma": lemma,
                    "speech_level": speech_level,
                }
                tracked.append(tracked_row)
                subjects = _explicit_subject_relations(
                    sentence=sentence,
                    predicate_token_id=token["token_id"],
                    persons=persons,
                )
                if not subjects:
                    role_default_attribution_count += 1
                    subject = _role_default_subject(
                        context=context,
                        speech_level=speech_level,
                    )
                    attribution = "role_default"
                elif len(subjects) == 1:
                    subject = subjects[0]
                    attribution = "explicit_subject"
                else:
                    unresolved.append(
                        {
                            **tracked_row,
                            "reason_code": "explicit_subject_relation_not_unique",
                            "candidate_count": len(subjects),
                        }
                    )
                    judge_candidates.append(
                        _judge_candidate(
                            turn_ref=turn["turn_ref"],
                            text=turn["text"],
                            predicate=token["text"],
                            question="この敬語の動作主を特定できますか。",
                        )
                    )
                    continue
                allowed = config[f"{speech_level}_allowed_relations"]
                if subject["relation"] not in allowed:
                    reason_code = f"{speech_level}_subject_relation_reversed"
                    violations.append(
                        {
                            "rule_id": f"KJ-DIRECTION-{speech_level.upper()}-001",
                            "source_ref": config["source_ref"],
                            "reason_code": reason_code,
                            "direction_reason_code": f"{speech_level}_subject_relation_reversed",
                            "attribution": attribution,
                            **tracked_row,
                            **subject,
                        }
                    )
    applicable = bool(tracked)
    measurable = applicable and not unresolved
    passed = not violations if measurable else None
    return _family_row(
        family="honorific_direction",
        applicable=applicable,
        measurable=measurable,
        passed=passed,
        violations=violations,
        diagnostics={
            "tracked_predicates": tracked,
            "unresolved_predicates": unresolved,
            "judge_candidates": judge_candidates,
            "role_default_attribution_count": role_default_attribution_count,
        },
    )


def _in_out_group_family(
    parsed: dict[str, Any],
    context: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    config = catalog["rule_families"]["in_out_group"]
    suffixes = config["honorific_suffixes"]
    occurrences: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    judge_candidates: list[dict[str, str]] = []
    violations: list[dict[str, Any]] = []
    surface_map: dict[str, list[dict[str, str]]] = {}
    for person in _person_surfaces(context):
        surface_map.setdefault(person["surface"], []).append(person)
    for turn in parsed["turns"]:
        for surface, persons in surface_map.items():
            for suffix in suffixes:
                pattern = re.escape(surface) + re.escape(suffix)
                for match in re.finditer(pattern, turn["text"]):
                    occurrence = {
                        **_span(
                            turn["turn_ref"],
                            match.start(),
                            match.end(),
                            match.group(0),
                        ),
                        "surface": surface,
                        "suffix": suffix,
                    }
                    occurrences.append(occurrence)
                    before = turn["text"][: match.start()]
                    if _customer_side_reference(before):
                        continue
                    if _operator_side_reference(before):
                        violations.append(
                            {
                                "rule_id": "KJ-INOUT-IN_GROUP_SUFFIX-001",
                                "source_ref": config["source_ref"],
                                "reason_code": "in_group_person_raised_to_external_customer",
                                **occurrence,
                            }
                        )
                        continue
                    if len(persons) != 1:
                        unresolved.append(
                            {
                                **occurrence,
                                "reason_code": "person_surface_relation_not_unique",
                            }
                        )
                        judge_candidates.append(
                            _judge_candidate(
                                turn_ref=turn["turn_ref"],
                                text=turn["text"],
                                predicate=match.group(0),
                                question="この人物がオペレーター組織側か顧客側かを特定できますか。",
                            )
                        )
                        continue
                    person = persons[0]
                    if person["relation"] in config["forbidden_relations"]:
                        violations.append(
                            {
                                "rule_id": "KJ-INOUT-IN_GROUP_SUFFIX-001",
                                "source_ref": config["source_ref"],
                                "reason_code": "in_group_person_raised_to_external_customer",
                                **occurrence,
                                **person,
                            }
                        )
    applicable = bool(occurrences)
    measurable = applicable and not unresolved
    passed = not violations if measurable else None
    return _family_row(
        family="in_out_group",
        applicable=applicable,
        measurable=measurable,
        passed=passed,
        violations=violations,
        diagnostics={
            "tracked_occurrences": occurrences,
            "unresolved_occurrences": unresolved,
            "judge_candidates": judge_candidates,
        },
    )


def _role_default_subject(
    *, context: dict[str, Any], speech_level: str
) -> dict[str, Any]:
    person_id = context["speaker_person_id"]
    relation = "self"
    if speech_level == "honorific":
        person_id = context["addressee_person_id"]
        relation = "customer"
    return {
        "subject_person_id": person_id,
        "subject_surface": None,
        "relation": relation,
        "subject_start_char": None,
        "subject_end_char": None,
    }


def _customer_side_reference(before: str) -> bool:
    return bool(re.search(r"(?:御社|貴社|お客様の|[ごお])$", before))


def _operator_side_reference(before: str) -> bool:
    return bool(re.search(r"(?:弊社|当社)(?:の担当)?の?$", before))


def _judge_candidate(
    *, turn_ref: str, text: str, predicate: str, question: str
) -> dict[str, str]:
    return {
        "turn_ref": turn_ref,
        "text": text,
        "predicate": predicate,
        "question": question,
    }


def _family_unresolved(family: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = family["diagnostics"]
    return diagnostics.get("unresolved_predicates", diagnostics.get("unresolved_occurrences", []))


def _explicit_subject_relations(
    *,
    sentence: dict[str, Any],
    predicate_token_id: int,
    persons: list[dict[str, str]],
) -> list[dict[str, Any]]:
    subject_tokens = [
        token
        for token in sentence["tokens"]
        if token["dep"] in {"nsubj", "nsubj:outer"}
        and token["head_token_id"] == predicate_token_id
    ]
    if not subject_tokens:
        return []
    sentence_start = sentence["start_char"]
    sentence_text = sentence["text"]
    candidates: list[dict[str, Any]] = []
    for person in persons:
        pattern = re.escape(person["surface"]) + r"(?:様)?(?=[がは])"
        for match in re.finditer(pattern, sentence_text):
            start = sentence_start + match.start()
            end = sentence_start + match.end()
            if not any(
                token["start_char"] < end and token["end_char"] > start
                for token in subject_tokens
            ):
                continue
            candidates.append(
                {
                    "subject_person_id": person["person_id"],
                    "subject_surface": match.group(0),
                    "relation": person["relation"],
                    "subject_start_char": start,
                    "subject_end_char": end,
                }
            )
    unique = {
        (
            item["subject_person_id"],
            item["subject_start_char"],
            item["subject_end_char"],
        ): item
        for item in candidates
    }
    return list(unique.values())


def _person_surfaces(context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "person_id": person["person_id"],
            "relation": person["relation_to_operator"],
            "surface": surface,
        }
        for person in context["persons"]
        for surface in person["surface_forms"]
    ]


def _family_row(
    *,
    family: str,
    applicable: bool,
    measurable: bool,
    passed: bool | None,
    violations: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "rule_family": family,
        "rule_applicable": applicable,
        "parser_measurable": measurable,
        "violation_count": len(violations),
        "passed": passed,
        "violations": violations,
        "diagnostics": diagnostics,
    }


def _span(turn_ref: str, start: int, end: int, text: str) -> dict[str, Any]:
    return {
        "turn_ref": turn_ref,
        "start_char": start,
        "end_char": end,
        "text": text,
    }


def _validate_rule_catalog_value(value: Any) -> dict[str, Any]:
    canonical = load_rule_catalog()
    if value != canonical:
        raise ValueError("in-memory Japanese rule catalog differs from the fixed catalog")
    return value


def _validate_parse_bundle(
    value: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"parse", "receipt"}:
        raise ValueError("Japanese parse bundle fields are invalid")
    parsed = value["parse"]
    receipt = value["receipt"]
    if not isinstance(parsed, dict) or parsed.get("schema_version") != PARSE_SCHEMA_VERSION:
        raise ValueError("Japanese parse schema version is invalid")
    if not isinstance(parsed.get("turns"), list):
        raise ValueError("Japanese parsed turns must be an array")
    if not isinstance(receipt, dict):
        raise ValueError("Japanese parse receipt must be an object")
    if receipt.get("parse_hash") != sha256_json(parsed):
        raise ValueError("Japanese parse receipt hash mismatch")
    receipt_body = {key: item for key, item in receipt.items() if key != "receipt_id"}
    if receipt.get("receipt_id") != sha256_json(receipt_body):
        raise ValueError("Japanese parse receipt ID mismatch")
    return parsed, receipt
