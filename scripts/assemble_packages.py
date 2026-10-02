#!/usr/bin/env python3
"""Join public task/solution JSON files into runner package JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-id", action="append")
    args = parser.parse_args()
    wanted = set(args.scenario_id or ())
    args.output.mkdir(parents=True, exist_ok=True)
    for task_path in sorted(args.tasks.glob("*.json")):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        scenario_id = task.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"{task_path}: scenario_id must be a non-empty string")
        if wanted and scenario_id not in wanted:
            continue
        solution_path = args.solutions / task_path.name
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
        overlap = set(task) & set(solution) - {"metadata"}
        if overlap:
            raise ValueError(f"{scenario_id}: task/solution fields overlap: {sorted(overlap)}")
        package = {**task, **solution}
        package.pop("metadata", None)
        (args.output / f"{scenario_id}.yaml").write_text(
            json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
