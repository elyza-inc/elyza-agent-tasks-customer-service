"""Content-addressed storage and deterministic replay for evaluator artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any


ARTIFACT_RECEIPT_SCHEMA_VERSION = "content_addressed_receipt"
SENSITIVE_KEY_FRAGMENTS = ("authorization", "api_key", "apikey", "access_token")
_REQUEST_BUNDLE_INDEX: dict[tuple[Path, str], dict[str, list[str]]] = {}
_REQUEST_BUNDLE_INDEX_LOCK = Lock()


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for any JSON-compatible value."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return a ``sha256:`` digest for a JSON-compatible value."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def assert_no_secrets(value: Any, *, path: str = "$") -> None:
    """Reject secret-bearing keys in a JSON request or response artifact."""

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS):
                raise ValueError(f"secret-bearing artifact key is forbidden: {path}.{key}")
            assert_no_secrets(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_secrets(item, path=f"{path}[{index}]")


def write_content_addressed_bundle(
    *,
    root: Path,
    namespace: str,
    request_payload: dict[str, Any],
    raw_response: dict[str, Any],
    parsed: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write an immutable JSON artifact bundle.

    Accepted inputs are JSON-compatible mappings. ``parsed`` may be ``None``
    when transport or parsing failed. Secret-bearing keys are invalid and
    raise ``ValueError``. An existing digest directory is verified byte for
    byte and is never overwritten.
    """

    if not namespace or "/" in namespace or namespace in {".", ".."}:
        raise ValueError(f"artifact namespace must be one path component: {namespace!r}")
    assert_no_secrets(request_payload)
    assert_no_secrets(raw_response)
    assert_no_secrets(parsed)
    assert_no_secrets(metadata)
    bundle_identity = {
        "request_hash": sha256_json(request_payload),
        "raw_response_hash": sha256_json(raw_response),
        "parsed_hash": sha256_json(parsed),
        "metadata_hash": sha256_json(metadata),
    }
    digest = hashlib.sha256(canonical_json_bytes(bundle_identity)).hexdigest()
    bundle_dir = root / namespace / digest
    receipt = {
        "schema_version": ARTIFACT_RECEIPT_SCHEMA_VERSION,
        "receipt_id": f"sha256:{digest}",
        "namespace": namespace,
        "bundle_identity": bundle_identity,
        "relative_path": f"{namespace}/{digest}",
        "metadata": metadata,
    }
    files = {
        "request.json": request_payload,
        "raw_response.json": raw_response,
        "parsed.json": parsed,
        "receipt.json": receipt,
    }
    if bundle_dir.exists():
        _verify_existing_bundle(bundle_dir=bundle_dir, expected=files)
        _register_request_bundle(root=root, namespace=namespace, request_payload=request_payload, bundle_name=digest)
        return receipt
    bundle_dir.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        _write_json(bundle_dir / name, value)
    _register_request_bundle(root=root, namespace=namespace, request_payload=request_payload, bundle_name=digest)
    return receipt


def replay_content_addressed_bundle(*, root: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    """Load and verify a previously stored bundle.

    The receipt must use ``content_addressed_receipt`` and point below
    ``root`` using the emitted relative path. Missing files, path traversal,
    malformed JSON, or any hash mismatch raise ``ValueError``.
    """

    if receipt.get("schema_version") != ARTIFACT_RECEIPT_SCHEMA_VERSION:
        raise ValueError(f"unsupported content-addressed receipt: {receipt.get('schema_version')}")
    relative_path = str(receipt.get("relative_path") or "")
    relative = Path(relative_path)
    if not relative_path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid content-addressed relative path: {relative_path!r}")
    bundle_dir = root / relative
    loaded = {
        "request": _read_json(bundle_dir / "request.json"),
        "raw_response": _read_json(bundle_dir / "raw_response.json"),
        "parsed": _read_json(bundle_dir / "parsed.json"),
        "receipt": _read_json(bundle_dir / "receipt.json"),
    }
    if loaded["receipt"] != receipt:
        raise ValueError("content-addressed receipt file does not match supplied receipt")
    expected_identity = {
        "request_hash": sha256_json(loaded["request"]),
        "raw_response_hash": sha256_json(loaded["raw_response"]),
        "parsed_hash": sha256_json(loaded["parsed"]),
        "metadata_hash": sha256_json(receipt.get("metadata")),
    }
    if receipt.get("bundle_identity") != expected_identity:
        raise ValueError("content-addressed bundle hash mismatch")
    digest = hashlib.sha256(canonical_json_bytes(expected_identity)).hexdigest()
    if receipt.get("receipt_id") != f"sha256:{digest}":
        raise ValueError("content-addressed receipt ID mismatch")
    if bundle_dir.name != digest:
        raise ValueError("content-addressed directory digest mismatch")
    return loaded


def find_content_addressed_receipt_by_request(
    *,
    root: Path,
    namespace: str,
    request_payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Find the single immutable receipt for an identical stored request.

    Missing namespaces return ``None``. Malformed bundles or more than one
    response for the same request raise ``ValueError`` so a caller cannot hide
    fresh-call nondeterminism behind an arbitrary cache choice.
    """

    namespace_dir = root / namespace
    if not namespace_dir.is_dir():
        return None
    bundle_names = _request_bundle_names(root=root, namespace=namespace, request_hash=sha256_json(request_payload))
    return _find_matching_receipt(
        root=root,
        request_payload=request_payload,
        bundle_dirs=[namespace_dir / name for name in bundle_names],
    )


def _request_bundle_names(*, root: Path, namespace: str, request_hash: str) -> list[str]:
    key = (root, namespace)
    with _REQUEST_BUNDLE_INDEX_LOCK:
        index = _REQUEST_BUNDLE_INDEX.get(key)
        if index is None:
            index = {}
            namespace_dir = root / namespace
            if namespace_dir.is_dir():
                skipped_count = 0
                skipped_example: Path | None = None
                for bundle_dir in sorted(namespace_dir.iterdir()):
                    if not bundle_dir.is_dir():
                        continue
                    try:
                        stored_request = _read_json(bundle_dir / "request.json")
                        stored_hash = sha256_json(stored_request)
                    except (TypeError, ValueError):
                        try:
                            receipt = _read_json(bundle_dir / "receipt.json")
                            stored_hash = receipt["bundle_identity"]["request_hash"]
                            if not isinstance(stored_hash, str):
                                raise ValueError("receipt request hash must be a string")
                        except (KeyError, TypeError, ValueError):
                            skipped_count += 1
                            skipped_example = skipped_example or bundle_dir
                            continue
                    index.setdefault(stored_hash, []).append(bundle_dir.name)
                if skipped_count:
                    logging.warning(
                        "Skipped %d unreadable content-addressed bundles while indexing; example: %s",
                        skipped_count,
                        skipped_example,
                    )
            _REQUEST_BUNDLE_INDEX[key] = index
        return list(index.get(request_hash, []))


def _register_request_bundle(
    *, root: Path, namespace: str, request_payload: dict[str, Any], bundle_name: str
) -> None:
    with _REQUEST_BUNDLE_INDEX_LOCK:
        index = _REQUEST_BUNDLE_INDEX.get((root, namespace))
        if index is None:
            return
        bundle_names = index.setdefault(sha256_json(request_payload), [])
        if bundle_name not in bundle_names:
            bundle_names.append(bundle_name)
            bundle_names.sort()


def _find_matching_receipt(
    *,
    root: Path,
    request_payload: dict[str, Any],
    bundle_dirs: list[Path],
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for bundle_dir in bundle_dirs:
        if not bundle_dir.is_dir():
            continue
        if _read_json(bundle_dir / "request.json") != request_payload:
            continue
        receipt = _read_json(bundle_dir / "receipt.json")
        if not isinstance(receipt, dict):
            raise ValueError(f"content-addressed receipt must be an object: {bundle_dir}")
        replay_content_addressed_bundle(root=root, receipt=receipt)
        matches.append(receipt)
    if len(matches) > 1:
        raise ValueError("multiple immutable responses exist for one identical request")
    return matches[0] if matches else None


def _verify_existing_bundle(*, bundle_dir: Path, expected: dict[str, Any]) -> None:
    for name, value in expected.items():
        path = bundle_dir / name
        if not path.is_file():
            raise ValueError(f"immutable artifact bundle is incomplete: {path}")
        if _read_json(path) != value:
            raise ValueError(f"immutable artifact collision: {path}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read content-addressed JSON artifact: {path}: {exc}") from exc
