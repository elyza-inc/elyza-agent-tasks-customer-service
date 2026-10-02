#!/usr/bin/env python3
"""実行+採点のエントリポイント。実体は src 配下の evaluation.cli.run_eval です。"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elyza_agent_tasks_customer_service.evaluation.cli.run_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
