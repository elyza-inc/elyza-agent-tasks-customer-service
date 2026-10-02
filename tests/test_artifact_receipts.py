"""Content-addressed receipt lookup checks."""

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts.artifact_receipts import (
    _request_bundle_names,
    find_content_addressed_receipt_by_request,
    sha256_json,
    write_content_addressed_bundle,
)


def _write_bundle(root, request, raw_response=None):
    return write_content_addressed_bundle(
        root=root,
        namespace="judge",
        request_payload=request,
        raw_response=raw_response or {"request": request},
        parsed={"ok": True},
        metadata={"call_kind": "test"},
    )


def test_request_lookup_indexes_existing_bundles_and_new_writes(tmp_path):
    requests = [{"id": index} for index in range(3)]
    receipts = [_write_bundle(tmp_path, request) for request in requests]

    assert [
        find_content_addressed_receipt_by_request(root=tmp_path, namespace="judge", request_payload=request)
        for request in requests
    ] == receipts
    assert find_content_addressed_receipt_by_request(root=tmp_path, namespace="judge", request_payload={"id": 9}) is None

    receipt = _write_bundle(tmp_path, {"id": 9})
    assert find_content_addressed_receipt_by_request(root=tmp_path, namespace="judge", request_payload={"id": 9}) == receipt


def test_request_index_uses_receipt_hash_when_request_file_is_missing(tmp_path, caplog):
    request = {"id": "missing-request"}
    receipt = _write_bundle(tmp_path, request)
    bundle_dir = tmp_path / receipt["relative_path"]
    (bundle_dir / "request.json").unlink()
    unreadable_bundle = tmp_path / "judge" / "unreadable"
    unreadable_bundle.mkdir()

    assert _request_bundle_names(root=tmp_path, namespace="judge", request_hash=sha256_json(request)) == [bundle_dir.name]
    assert "Skipped 1 unreadable content-addressed bundles" in caplog.text
    assert str(unreadable_bundle) in caplog.text


def test_request_lookup_rejects_missing_request_file_for_matching_payload(tmp_path):
    request = {"id": "restore-on-match"}
    receipt = _write_bundle(tmp_path, request)
    request_path = tmp_path / receipt["relative_path"] / "request.json"
    request_path.unlink()

    with pytest.raises(ValueError, match="cannot read content-addressed JSON artifact"):
        find_content_addressed_receipt_by_request(root=tmp_path, namespace="judge", request_payload=request)
    assert not request_path.exists()


def test_request_lookup_does_not_restore_missing_request_file_for_different_payload(tmp_path):
    receipt = _write_bundle(tmp_path, {"id": "stored"})
    request_path = tmp_path / receipt["relative_path"] / "request.json"
    request_path.unlink()

    assert find_content_addressed_receipt_by_request(
        root=tmp_path,
        namespace="judge",
        request_payload={"id": "different"},
    ) is None
    assert not request_path.exists()


