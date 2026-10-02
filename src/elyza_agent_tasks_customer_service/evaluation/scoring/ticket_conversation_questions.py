"""Generate closed questions from operator-authored ticket fields."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.config_paths import config_path
from elyza_agent_tasks_customer_service.evaluation.contracts.artifact_receipts import sha256_json


DEFAULT_TEMPLATE_PATH = (
    config_path("ticket_conversation_question_templates.json")
)
QUESTION_VERSION = "ticket_conversation_questions"
TEMPLATE_VERSION = "ticket_conversation_question_templates"
TEMPLATE_IDS = (
    "refusal_reason_code",
    "promise_callback",
    "answer_summary",
)
MAX_YES_QUESTIONS = 3
DECOY_QUESTION_COUNT = 2
DISCLOSURE_QUESTION = "オペレータは、契約上必要な案内「{disclosure}」を会話中に伝えたか。"
MUTATION_QUESTION = "オペレータは、{operation}を実施したと会話中に伝えたか。"
SOURCE_SOLUTIONS_ROOT = Path(__file__).resolve().parents[4] / "data" / "solutions"
CONSENT_QUESTION = (
    "会話中に、次の同意についてオペレーターが顧客に確認し、顧客から同意の返事を得たか。"
    "言い回しの違いは問わない: 「{consent}」"
)


def load_ticket_question_templates(
    path: Path = DEFAULT_TEMPLATE_PATH,
) -> dict[str, Any]:
    """Load an object-root JSON template file.

    The file must use ``ticket_conversation_question_templates`` and contain
    the three fixed templates in ``TEMPLATE_IDS`` order. Non-JSON files,
    malformed JSON, unknown fields, and invalid template rows raise
    ``ValueError``.
    """

    if path.suffix.lower() != ".json":
        raise ValueError(f"ticket question templates must be JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load ticket question templates {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "templates"}:
        raise ValueError("ticket question template root fields are invalid")
    if value.get("schema_version") != TEMPLATE_VERSION:
        raise ValueError("unsupported ticket question template version")
    templates = value.get("templates")
    if not isinstance(templates, list):
        raise ValueError("ticket question templates must be an array")
    ids = tuple(item.get("template_id") for item in templates if isinstance(item, dict))
    if ids != TEMPLATE_IDS:
        raise ValueError(f"ticket question template IDs mismatch: {ids}")
    for index, template in enumerate(templates):
        required = {"template_id", "question_text", "required_parameters"}
        if not isinstance(template, dict) or set(template) != required:
            raise ValueError(f"ticket question templates[{index}] fields are invalid")
        if not isinstance(template["question_text"], str) or not template["question_text"]:
            raise ValueError(f"ticket question templates[{index}].question_text is invalid")
        parameters = template["required_parameters"]
        if not isinstance(parameters, list) or not all(
            isinstance(item, str) and item for item in parameters
        ):
            raise ValueError(
                f"ticket question templates[{index}].required_parameters is invalid"
            )
    return value


def build_ticket_conversation_questions(
    ticket: dict[str, Any],
    *,
    candidate_span_refs: list[str] | None = None,
    code_descriptions: dict[str, str] | None = None,
    scenario_id: str | None = None,
    domain_id: str | None = None,
    completion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic yes/no questions for one ticket and completion contract.

    ``ticket`` must be an object using operator-ticket field names.
    ``candidate_span_refs`` may contain frozen conversation-event refs; when
    omitted, an empty list is emitted for the later judge orchestration step.
    ``code_descriptions`` optionally maps typed identifiers to their public
    descriptions. ``completion`` accepts the completion-contract fields
    ``required_disclosures`` (a string array), ``required_consent`` (a string
    or null), and ``forbidden_mutations`` (an object array). When supplied,
    it fills up to three expected-yes questions and supplies expected-no
    decoys. Peer-scenario disclosures are read from the local solution set for
    ``domain_id``. Malformed populated values raise ``ValueError``.
    """

    if not isinstance(ticket, dict):
        raise ValueError("ticket must be an object")
    spans = [] if candidate_span_refs is None else candidate_span_refs
    if not isinstance(spans, list) or not all(isinstance(item, str) and item for item in spans):
        raise ValueError("candidate_span_refs must be an array of non-empty strings")
    descriptions = {} if code_descriptions is None else code_descriptions
    if not isinstance(descriptions, dict) or not all(
        isinstance(code, str)
        and code
        and isinstance(description, str)
        and description
        for code, description in descriptions.items()
    ):
        raise ValueError("code_descriptions must map strings to non-empty strings or be null")
    if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
        raise ValueError("scenario_id must be a non-empty string or null")
    if domain_id is not None and (not isinstance(domain_id, str) or not domain_id):
        raise ValueError("domain_id must be a non-empty string or null")
    contract = _completion_contract(completion)
    peer_disclosures = (
        [] if domain_id is None
        else list(dict.fromkeys(_source_domain_disclosures(domain_id, scenario_id)))
    )
    templates_value = load_ticket_question_templates()
    templates = {item["template_id"]: item for item in templates_value["templates"]}
    questions: list[dict[str, Any]] = []
    if "refusal_reason" in ticket:
        refusal_field = "refusal_reason"
    else:
        refusal_field = "refusal_reason_code"
    refusal_reason = ticket.get(refusal_field)
    if refusal_reason is not None:
        _require_scalar(refusal_reason, "refusal_reason_code")
        questions.append(
            _question(
                question_id="ticket:refusal_reason_code",
                template=templates["refusal_reason_code"],
                parameters={"reason": descriptions.get(refusal_reason, refusal_reason)},
                source_ref=f"ticket:{refusal_field}",
                candidate_span_refs=spans,
            )
        )
    promises = ticket.get("promises") or []
    if not isinstance(promises, list):
        raise ValueError("ticket.promises must be an array")
    for index, promise in enumerate(promises):
        if not isinstance(promise, dict):
            raise ValueError(f"ticket.promises[{index}] must be an object")
        owner = promise.get("owner_id")
        channel = promise.get("channel")
        deadline = promise.get("deadline")
        _require_scalar(owner, f"promises[{index}].owner_id")
        _require_scalar(channel, f"promises[{index}].channel")
        _require_scalar(deadline, f"promises[{index}].deadline")
        questions.append(
            _question(
                question_id=f"ticket:promises[{index}]:callback",
                template=templates["promise_callback"],
                parameters={"owner": owner, "channel": channel, "deadline": deadline},
                source_ref=f"ticket:promises[{index}]",
                candidate_span_refs=spans,
            )
        )
    answer_summary = ticket.get("answer_summary")
    if answer_summary is not None:
        summary = _answer_summary_text(answer_summary, descriptions)
        questions.append(
            _question(
                question_id="ticket:answer_summary",
                template=templates["answer_summary"],
                parameters={"summary": summary},
                source_ref="ticket:answer_summary",
                candidate_span_refs=spans,
            )
        )
    yes_questions = questions[:MAX_YES_QUESTIONS]
    known_disclosures = {
        value
        for question in yes_questions
        for value in question["parameters"].values()
        if isinstance(value, str)
    }
    for index, disclosure in enumerate(contract["required_disclosures"]):
        if len(yes_questions) == MAX_YES_QUESTIONS:
            break
        known_disclosures.add(disclosure)
        yes_questions.append(
            _contract_question(
                question_id=f"contract:required_disclosures[{index}]",
                template_id="required_disclosure",
                question_text=DISCLOSURE_QUESTION.format(disclosure=disclosure),
                source_ref=f"contracts.completion.required_disclosures[{index}]",
                candidate_span_refs=spans,
            )
        )
    consent = contract["required_consent"]
    if len(yes_questions) < MAX_YES_QUESTIONS and consent is not None:
        known_disclosures.add(consent)
        yes_questions.append(
            _contract_question(
                question_id="contract:required_consent",
                template_id="required_consent",
                question_text=CONSENT_QUESTION.format(consent=consent),
                source_ref="contracts.completion.required_consent",
                candidate_span_refs=spans,
            )
        )
    decoys = _decoy_questions(
        scenario_id=scenario_id or "ticket",
        candidate_span_refs=spans,
        known_disclosures=known_disclosures,
        peer_disclosures=peer_disclosures,
        forbidden_mutations=contract["forbidden_mutations"],
    )
    for question in yes_questions + decoys:
        question.pop("parameters", None)
    return {
        "schema_version": QUESTION_VERSION,
        "template_version": templates_value["schema_version"],
        "template_hash": sha256_json(templates_value),
        "questions": yes_questions + decoys,
    }


def _question(
    *,
    question_id: str,
    template: dict[str, Any],
    parameters: dict[str, Any],
    source_ref: str,
    candidate_span_refs: list[str],
) -> dict[str, Any]:
    missing = set(template["required_parameters"]) - set(parameters)
    if missing:
        raise ValueError(f"ticket question parameters missing: {sorted(missing)}")
    try:
        question_text = template["question_text"].format(**parameters)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"ticket question template rendering failed: {exc}") from exc
    return {
        "question_id": question_id,
        "metric_id": "M15",
        "template_id": template["template_id"],
        "question_text": question_text,
        "candidate_span_refs": list(candidate_span_refs),
        "source_refs": [source_ref],
        "required_evidence_actors": ["operator"],
        "expected_answer": "yes",
        "parameters": parameters,
    }


def _contract_question(
    *,
    question_id: str,
    template_id: str,
    question_text: str,
    source_ref: str,
    candidate_span_refs: list[str],
) -> dict[str, Any]:
    """Build one contract-derived question with no private judge context."""

    return {
        "question_id": question_id,
        "metric_id": "M15",
        "template_id": template_id,
        "question_text": question_text,
        "candidate_span_refs": list(candidate_span_refs),
        "source_refs": [source_ref],
        "required_evidence_actors": ["operator"],
        "expected_answer": "yes",
        "parameters": {"disclosure": question_text},
    }


def _decoy_questions(
    *,
    scenario_id: str,
    candidate_span_refs: list[str],
    known_disclosures: set[str],
    peer_disclosures: list[str],
    forbidden_mutations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return at most two deterministic, non-overlapping expected-no questions."""

    questions = []
    candidate = _first_non_overlapping(peer_disclosures, known_disclosures, scenario_id)
    if candidate is not None:
        questions.append(
            {
                **_contract_question(
                    question_id=f"decoy:peer_disclosure:{_short_hash(candidate)}",
                    template_id="peer_required_disclosure",
                    question_text=DISCLOSURE_QUESTION.format(disclosure=candidate),
                    source_ref="same_domain.required_disclosures",
                    candidate_span_refs=candidate_span_refs,
                ),
                "expected_answer": "no",
            }
        )
    for index, mutation in enumerate(forbidden_mutations):
        if len(questions) == DECOY_QUESTION_COUNT:
            break
        operation = _mutation_description(mutation)
        questions.append(
            {
                **_contract_question(
                    question_id=f"decoy:forbidden_mutations[{index}]",
                    template_id="forbidden_mutation",
                    question_text=MUTATION_QUESTION.format(operation=operation),
                    source_ref=f"contracts.completion.forbidden_mutations[{index}]",
                    candidate_span_refs=candidate_span_refs,
                ),
                "expected_answer": "no",
            }
        )
    return questions


def _completion_contract(completion: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the completion fields used to construct M15 questions."""

    if completion is None:
        return {"required_disclosures": [], "required_consent": None, "forbidden_mutations": []}
    if not isinstance(completion, dict):
        raise ValueError("completion must be an object or null")
    disclosures = completion.get("required_disclosures", [])
    consent = completion.get("required_consent")
    mutations = completion.get("forbidden_mutations", [])
    if not isinstance(disclosures, list) or not all(isinstance(item, str) and item for item in disclosures):
        raise ValueError("completion.required_disclosures must be an array of non-empty strings")
    if consent is not None and (not isinstance(consent, str) or not consent):
        raise ValueError("completion.required_consent must be a non-empty string or null")
    if not isinstance(mutations, list) or not all(isinstance(item, dict) for item in mutations):
        raise ValueError("completion.forbidden_mutations must be an object array")
    return {
        "required_disclosures": disclosures,
        "required_consent": consent,
        "forbidden_mutations": mutations,
    }


def _source_domain_disclosures(domain_id: str, scenario_id: str | None) -> list[str]:
    """Read peer ``required_disclosures`` values from checked-in solutions."""

    root = SOURCE_SOLUTIONS_ROOT / domain_id
    if not root.is_dir():
        return []
    disclosures = []
    for path in sorted(root.glob("*.json")):
        if path.stem == scenario_id:
            continue
        try:
            solution = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load peer solution {path}: {exc}") from exc
        contracts = solution.get("contracts") if isinstance(solution, dict) else None
        completion = contracts.get("completion") if isinstance(contracts, dict) else None
        if not isinstance(completion, dict):
            continue
        values = completion.get("required_disclosures", [])
        if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
            raise ValueError(f"{path}: completion.required_disclosures must be an array of non-empty strings")
        disclosures.extend(values)
    return disclosures


def _first_non_overlapping(
    candidates: list[str], known_disclosures: set[str], scenario_id: str
) -> str | None:
    """Choose the scenario-hash offset candidate that is not a correct disclosure."""

    ordered = sorted(candidates)
    if not ordered:
        return None
    start = int(sha256(scenario_id.encode("utf-8")).hexdigest(), 16) % len(ordered)
    for offset in range(len(ordered)):
        candidate = ordered[(start + offset) % len(ordered)]
        if candidate not in known_disclosures:
            return candidate
    return None


def _mutation_description(mutation: dict[str, Any]) -> str:
    """Render one forbidden mutation as a stable public operation description."""

    table = mutation.get("table")
    if not isinstance(table, str) or not table:
        raise ValueError("completion.forbidden_mutations[].table must be a non-empty string")
    kind = mutation.get("kind", "update")
    if not isinstance(kind, str) or not kind:
        raise ValueError("completion.forbidden_mutations[].kind must be a non-empty string")
    return f"{table} に対する {kind} 操作（{json.dumps(mutation, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}）"


def _short_hash(value: str) -> str:
    """Return a stable compact identifier for a disclosure-derived decoy."""

    return sha256(value.encode("utf-8")).hexdigest()[:12]


def _answer_summary_text(value: Any, code_descriptions: dict[str, str]) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        for key in ("summary", "diagnostic_conclusion_code"):
            text = value.get(key)
            if isinstance(text, str) and text:
                return code_descriptions.get(text, text)
    raise ValueError("ticket.answer_summary has no question-ready summary text")


def _require_scalar(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"ticket.{label} must be a non-empty string")
