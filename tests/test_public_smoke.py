"""オフラインの疎通テスト。API キーなしで実行できます。"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_package_imports():
    from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import (
        load_metric_inventory,
    )

    inventory = load_metric_inventory()
    assert inventory


def test_assemble_and_convert(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "assemble_packages.py"),
            "--tasks", str(ROOT / "data" / "tasks" / "hotel"),
            "--solutions", str(ROOT / "data" / "solutions" / "hotel"),
            "--scenario-id", "htl-001",
            "--output", str(tmp_path),
        ],
        check=True,
    )
    from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import (
        convert_package,
        load_package,
    )

    package = load_package(tmp_path / "htl-001.yaml")
    assert convert_package(package, mode="text")
    assert convert_package(package, mode="audio-audio")


def test_assemble_writes_scenario_id_filename(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "assemble_packages.py"),
            "--tasks", str(ROOT / "data" / "tasks" / "hotel"),
            "--solutions", str(ROOT / "data" / "solutions" / "hotel"),
            "--scenario-id", "htl-020",
            "--output", str(tmp_path),
        ],
        check=True,
    )
    assert (tmp_path / "htl-020.yaml").is_file()
