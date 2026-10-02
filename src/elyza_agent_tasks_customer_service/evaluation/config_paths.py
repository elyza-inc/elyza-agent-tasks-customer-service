"""Resolve bundled configuration files.

``configs/`` はリポジトリ直下に置く。パッケージデータには含めないため、
ソースツリーから (``PYTHONPATH=src`` または editable install で) 実行したときだけ解決できる。
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def config_path(name: str) -> Path:
    """Return ``<repo>/configs/<name>``; raise ``FileNotFoundError`` if it is missing."""

    candidate = CONFIG_DIR / name
    if not candidate.is_file():
        raise FileNotFoundError(f"configs/{name} not found below: {CONFIG_DIR}")
    return candidate
