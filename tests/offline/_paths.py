"""Shared paths for the legacy offline suites."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
