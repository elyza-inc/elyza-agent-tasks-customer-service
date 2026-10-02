"""Regression test for the CI network guard."""

import os
import socket

import pytest


@pytest.mark.skipif(os.getenv("CI_NO_NETWORK") != "1", reason="CI network guard is disabled")
def test_ci_network_guard_blocks_external_connections() -> None:
    with socket.socket() as client, pytest.raises(RuntimeError, match="network connection blocked"):
        client.connect(("198.51.100.1", 443))
