#!/usr/bin/env python3
"""record 1件の後からの再採点。実体は src 配下の evaluation.scoring.package_scoring です。"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import main

if __name__ == "__main__":
    raise SystemExit(main())
