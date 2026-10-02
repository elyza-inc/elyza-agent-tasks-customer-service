"""Pinned GiNZA adapter producing replayable Japanese parse receipts."""

from __future__ import annotations

from elyza_agent_tasks_customer_service.evaluation.config_paths import config_path
import hashlib
import json
from functools import lru_cache
import threading
from importlib import metadata, util
from pathlib import Path
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.contracts.artifact_receipts import sha256_json


DEFAULT_PARSER_LOCK_PATH = config_path("japanese_parser_lock.json")
PARSER_LOCK_VERSION = "japanese_parser_lock"
# GiNZA が内部で使う SudachiPy は Rust 実装で再入できないため、解析呼び出しを直列化する。
_PARSE_LOCK = threading.Lock()
PARSE_SCHEMA_VERSION = "japanese_dependency_parse"
PARSE_RECEIPT_VERSION = "japanese_parse_receipt"
PARSE_ATTEMPT_VERSION = "japanese_parse_attempt"
ABI_ERROR_MARKERS = (
    "numpy.dtype size changed",
    "binary incompatibility",
    "undefined symbol",
    "symbol not found",
    "dlopen",
)
MISSING_MODEL_ERROR_MARKERS = (
    "[e050]",
    "can't find model",
    "is not a python package or a valid path to a data directory",
)
PINNED_PARSER_MODULE_PREFIXES = (
    "ginza",
    "ja_ginza",
    "spacy",
    "numpy",
    "thinc",
    "sudachipy",
    "sudachidict_core",
)


class JapaneseParserDependencyError(ValueError):
    """Raised when the pinned M17 parser environment cannot start."""


def load_parser_lock(path: Path = DEFAULT_PARSER_LOCK_PATH) -> dict[str, Any]:
    """Load the fixed parser lock from an object-root ``.json`` file.

    Arrays/scalars, malformed JSON, unknown/missing fields, unsupported parser
    options, and non-SHA resource hashes raise ``ValueError``.
    """

    if path.suffix.lower() != ".json":
        raise ValueError(f"Japanese parser lock must be JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load Japanese parser lock {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Japanese parser lock must be an object")
    required = {
        "schema_version",
        "model_name",
        "distributions",
        "resource_hashes",
        "parser_options",
    }
    if set(value) != required:
        raise ValueError("Japanese parser lock fields are invalid")
    if value["schema_version"] != PARSER_LOCK_VERSION:
        raise ValueError("unsupported Japanese parser lock")
    if value["model_name"] != "ja_ginza":
        raise ValueError("Japanese parser model must be ja_ginza")
    expected_distributions = {
        "ginza",
        "ja-ginza",
        "spacy",
        "numpy",
        "thinc",
        "SudachiPy",
        "SudachiDict-core",
    }
    distributions = value["distributions"]
    if not isinstance(distributions, dict) or set(distributions) != expected_distributions:
        raise ValueError("Japanese parser distribution lock is incomplete")
    if not all(isinstance(version, str) and version for version in distributions.values()):
        raise ValueError("Japanese parser versions must be non-empty strings")
    expected_resources = {
        "ginza_python_tree",
        "ja_ginza_model_tree",
        "sudachi_system_dictionary",
    }
    hashes = value["resource_hashes"]
    if not isinstance(hashes, dict) or set(hashes) != expected_resources:
        raise ValueError("Japanese parser resource hash lock is incomplete")
    if not all(
        isinstance(digest, str)
        and digest.startswith("sha256:")
        and len(digest) == 71
        for digest in hashes.values()
    ):
        raise ValueError("Japanese parser resource hashes must be SHA-256 digests")
    expected_options = {
        "exclude_pipeline_components",
        "gpu",
        "process_count",
        "sudachi_split_mode",
    }
    options = value["parser_options"]
    if not isinstance(options, dict) or set(options) != expected_options:
        raise ValueError("Japanese parser options are invalid")
    if options != {
        "exclude_pipeline_components": ["ner"],
        "gpu": False,
        "process_count": 1,
        "sudachi_split_mode": "C",
    }:
        raise ValueError("Japanese parser options drifted from the fixed CPU configuration")
    return value


def verify_pinned_parser_resources(
    lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify exact package versions and model/dictionary byte hashes."""

    if lock is None:
        lock = load_parser_lock()
    else:
        load_parser_lock_value(lock)
    observed_versions: dict[str, str] = {}
    for distribution, expected_version in lock["distributions"].items():
        try:
            observed_version = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise JapaneseParserDependencyError(
                f"Japanese parser distribution is missing: {distribution}"
            ) from exc
        if observed_version != expected_version:
            raise JapaneseParserDependencyError(
                f"Japanese parser version drift: {distribution} "
                f"expected={expected_version} observed={observed_version}"
            )
        observed_versions[distribution] = observed_version
    try:
        resource_paths = _resource_paths()
        observed_hashes = {
            "ginza_python_tree": _hash_tree(resource_paths["ginza_python_tree"]),
            "ja_ginza_model_tree": _hash_tree(resource_paths["ja_ginza_model_tree"]),
            "sudachi_system_dictionary": _hash_file(
                resource_paths["sudachi_system_dictionary"]
            ),
        }
    except ValueError as exc:
        if not _is_pinned_resource_unavailable_error(exc):
            raise
        raise JapaneseParserDependencyError(
            f"Japanese parser pinned resource is unavailable: {exc}"
        ) from exc
    if observed_hashes != lock["resource_hashes"]:
        raise JapaneseParserDependencyError(
            "Japanese parser resource hash drift: "
            f"expected={lock['resource_hashes']} observed={observed_hashes}"
        )
    return {
        "distribution_versions": observed_versions,
        "resource_hashes": observed_hashes,
        "parser_lock_hash": sha256_json(lock),
    }


def parse_japanese_turns(
    *,
    turns: list[dict[str, str]],
    lock_path: Path = DEFAULT_PARSER_LOCK_PATH,
) -> dict[str, Any]:
    """Parse operator turns and return fixed tokens/dependencies plus a receipt.

    ``turns`` must be an array of exact objects with non-empty ``turn_ref`` and
    string ``text`` fields. Duplicate refs, unknown fields, missing packages,
    version/hash drift, and parser failures raise ``ValueError``. No raw spaCy
    objects are returned or required for replay.
    """

    _validate_turns(turns)
    lock = load_parser_lock(lock_path)
    resource_receipt = verify_pinned_parser_resources(lock)
    nlp = _load_ginza(
        lock["model_name"],
        tuple(lock["parser_options"]["exclude_pipeline_components"]),
    )
    parsed_turns: list[dict[str, Any]] = []
    for turn in turns:
        with _PARSE_LOCK:
            doc = nlp(turn["text"])
        sentences: list[dict[str, Any]] = []
        for sentence_id, sentence in enumerate(doc.sents):
            tokens = []
            for token in sentence:
                tokens.append(
                    {
                        "token_id": token.i,
                        "text": token.text,
                        "lemma": token.lemma_,
                        "norm": token.norm_,
                        "pos": token.pos_,
                        "tag": token.tag_,
                        "dep": token.dep_,
                        "head_token_id": token.head.i,
                        "start_char": token.idx,
                        "end_char": token.idx + len(token.text),
                        "morph": sorted(str(token.morph).split("|"))
                        if str(token.morph)
                        else [],
                    }
                )
            sentences.append(
                {
                    "sentence_id": sentence_id,
                    "start_char": sentence.start_char,
                    "end_char": sentence.end_char,
                    "text": sentence.text,
                    "tokens": tokens,
                }
            )
        parsed_turns.append(
            {
                "turn_ref": turn["turn_ref"],
                "text": turn["text"],
                "sentences": sentences,
            }
        )
    parsed = {
        "schema_version": PARSE_SCHEMA_VERSION,
        "turns": parsed_turns,
    }
    receipt_body = {
        "schema_version": PARSE_RECEIPT_VERSION,
        **resource_receipt,
        "model_name": lock["model_name"],
        "parser_options": lock["parser_options"],
        "adapter_source_hash": _hash_file(Path(__file__)),
        "input_hash": sha256_json(turns),
        "parse_hash": sha256_json(parsed),
    }
    return {
        "parse": parsed,
        "receipt": {
            **receipt_body,
            "receipt_id": sha256_json(receipt_body),
        },
    }


def parse_japanese_turns_or_nm(
    *,
    turns: list[dict[str, str]],
    lock_path: Path = DEFAULT_PARSER_LOCK_PATH,
) -> dict[str, Any]:
    """Return a parse attempt without crashing on dependency unavailability.

    ``turns`` accepts the same exact object array as ``parse_japanese_turns``.
    Invalid turn or lock contracts still raise ``ValueError``. Missing pinned
    distributions, version/ABI drift, and model startup failures return an
    object with ``status=N/M`` and no parse bundle so an episode runner can
    persist the measurement state instead of terminating the whole run.
    """

    try:
        bundle = parse_japanese_turns(turns=turns, lock_path=lock_path)
    except Exception as exc:  # noqa: BLE001 - classified dependency boundary.
        if not is_japanese_parser_dependency_error(exc):
            raise
        return {
            "schema_version": PARSE_ATTEMPT_VERSION,
            "status": "N/M",
            "reason": "pinned_japanese_parser_dependency_unavailable",
            "parse_bundle": None,
            "diagnostics": {
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        }
    return {
        "schema_version": PARSE_ATTEMPT_VERSION,
        "status": "parsed",
        "reason": "pinned_japanese_parser_ready",
        "parse_bundle": bundle,
        "diagnostics": {},
    }


def is_japanese_parser_dependency_error(exc: Exception) -> bool:
    """Return whether an exception represents M17 environment unavailability."""

    return isinstance(exc, JapaneseParserDependencyError)


def load_parser_lock_value(value: Any) -> dict[str, Any]:
    """Validate an in-memory parser lock using the on-disk lock contract."""

    if not isinstance(value, dict):
        raise ValueError("Japanese parser lock must be an object")
    canonical = load_parser_lock()
    if set(value) != set(canonical):
        raise ValueError("Japanese parser lock fields are invalid")
    if value != canonical:
        raise ValueError("in-memory Japanese parser lock differs from the fixed lock")
    return value


@lru_cache(maxsize=1)
def _load_ginza(model_name: str, excluded: tuple[str, ...]) -> Any:
    try:
        import ginza
        import spacy
    except Exception as exc:  # noqa: BLE001 - dependency startup is classified below.
        if not _is_known_dependency_startup_error(exc):
            raise
        raise JapaneseParserDependencyError(
            "pinned spaCy/GiNZA packages are unavailable"
        ) from exc
    try:
        spacy.require_cpu()
        nlp = spacy.load(model_name, exclude=list(excluded))
    except Exception as exc:  # noqa: BLE001 - model/dependency startup boundary.
        if not _is_known_dependency_startup_error(exc, allow_missing_model=True):
            raise
        raise JapaneseParserDependencyError(
            f"cannot load pinned Japanese parser model {model_name}: {exc}"
        ) from exc
    ginza.set_split_mode(nlp, "C")
    return nlp


def _is_known_dependency_startup_error(
    exc: Exception,
    *,
    allow_missing_model: bool = False,
) -> bool:
    """Classify only pinned-package absence, ABI failure, or model absence."""

    module_name = getattr(exc, "name", None)
    if isinstance(exc, (ModuleNotFoundError, ImportError)) and isinstance(
        module_name, str
    ):
        normalized = module_name.lower()
        if any(
            normalized == prefix or normalized.startswith(prefix + ".")
            for prefix in PINNED_PARSER_MODULE_PREFIXES
        ):
            return True
    message = str(exc).lower()
    if any(marker in message for marker in ABI_ERROR_MARKERS):
        return True
    if allow_missing_model and isinstance(exc, OSError):
        return any(marker in message for marker in MISSING_MODEL_ERROR_MARKERS)
    return False


def _is_pinned_resource_unavailable_error(exc: ValueError) -> bool:
    message = str(exc)
    return message.startswith(
        (
            "Japanese parser resource directory is missing:",
            "Japanese parser resource directory is empty:",
            "cannot read Japanese parser resource ",
        )
    )


def _resource_paths() -> dict[str, Path]:
    ginza_root = _module_root("ginza")
    ja_ginza_root = _module_root("ja_ginza") / "ja_ginza-5.2.0"
    dictionary_path = (
        _module_root("sudachidict_core") / "resources" / "system.dic"
    )
    return {
        "ginza_python_tree": ginza_root,
        "ja_ginza_model_tree": ja_ginza_root,
        "sudachi_system_dictionary": dictionary_path,
    }


def _module_root(module_name: str) -> Path:
    spec = util.find_spec(module_name)
    if spec is None or not spec.submodule_search_locations:
        raise JapaneseParserDependencyError(
            f"Japanese parser module is missing: {module_name}"
        )
    return Path(next(iter(spec.submodule_search_locations)))


def _hash_tree(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"Japanese parser resource directory is missing: {path}")
    digest = hashlib.sha256()
    file_count = 0
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if "__pycache__" in candidate.parts or candidate.suffix == ".pyc":
            continue
        relative = candidate.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
        file_count += 1
    if not file_count:
        raise ValueError(f"Japanese parser resource directory is empty: {path}")
    return "sha256:" + digest.hexdigest()


def _hash_file(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read Japanese parser resource {path}: {exc}") from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _validate_turns(turns: Any) -> None:
    if not isinstance(turns, list):
        raise ValueError("Japanese parser turns must be an array")
    refs: list[str] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict) or set(turn) != {"turn_ref", "text"}:
            raise ValueError(f"Japanese parser turns[{index}] fields are invalid")
        if not isinstance(turn["turn_ref"], str) or not turn["turn_ref"]:
            raise ValueError(f"Japanese parser turns[{index}].turn_ref is invalid")
        if not isinstance(turn["text"], str):
            raise ValueError(f"Japanese parser turns[{index}].text must be a string")
        refs.append(turn["turn_ref"])
    if len(refs) != len(set(refs)):
        raise ValueError("Japanese parser turn_ref values must be unique")
