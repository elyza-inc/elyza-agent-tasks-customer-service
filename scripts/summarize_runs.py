#!/usr/bin/env python3
"""score.json の集計(参考値の表の再現)。実体は src 配下の evaluation.aggregate.summarize_runs です。"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elyza_agent_tasks_customer_service.evaluation.aggregate.summarize_runs import main

if __name__ == "__main__":
    raise SystemExit(main())
