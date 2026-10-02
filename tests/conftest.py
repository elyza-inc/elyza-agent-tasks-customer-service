"""Shared pytest setup for generated evaluation packages."""

from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import sys
import shutil
from pathlib import Path

import pytest

from tests import SCENARIO_COUNT


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PACKAGES_DIR = ROOT / ".run" / "packages"
PACKAGE_SOURCES = (
    (ROOT / "data" / "tasks", ROOT / "data" / "solutions"),
)
ASSEMBLE_SCRIPT = ROOT / "scripts" / "assemble_packages.py"
CI_NO_NETWORK = "CI_NO_NETWORK"


def _is_local_connection(address: object) -> bool:
    """Return whether a socket address is a Unix socket or loopback host."""
    if isinstance(address, (str, bytes)):
        return True
    if not isinstance(address, tuple) or not address:
        return False

    host = address[0]
    if host in ("localhost", b"localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped_ipv4 = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or (mapped_ipv4 is not None and mapped_ipv4.is_loopback)


@pytest.fixture(scope="session", autouse=True)
def block_network_in_ci() -> None:
    """Reject non-local socket connections when CI_NO_NETWORK=1."""
    if os.getenv(CI_NO_NETWORK) != "1":
        yield
        return

    original_connect = socket.socket.connect

    def connect(sock: socket.socket, address: object) -> None:
        if not _is_local_connection(address):
            raise RuntimeError(f"network connection blocked by {CI_NO_NETWORK}: {address!r}")
        original_connect(sock, address)

    socket.socket.connect = connect
    try:
        yield
    finally:
        socket.socket.connect = original_connect


@pytest.fixture(scope="session", autouse=True)
def assemble_packages() -> None:
    """Build all public and private packages into the sole generated directory."""
    shutil.rmtree(PACKAGES_DIR, ignore_errors=True)
    for tasks_root, solutions_root in PACKAGE_SOURCES:
        for tasks_dir in sorted(tasks_root.iterdir()):
            if tasks_dir.is_dir():
                subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLE_SCRIPT),
                    "--tasks",
                    str(tasks_dir),
                    "--solutions",
                    str(solutions_root / tasks_dir.name),
                    "--output",
                    str(PACKAGES_DIR),
                ],
                check=True,
            )
    assert len(list(PACKAGES_DIR.glob("*.yaml"))) == SCENARIO_COUNT
