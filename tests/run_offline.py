#!/usr/bin/env python3
"""Run the offline suites from the repository root.

They predate unittest and use their own ``check()`` counters, so they stay as
scripts under ``tests/offline``. The runner keeps cwd at the repository root
and reports every failing suite before exiting nonzero.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OFFLINE = HERE / "offline"


def suites():
    return sorted(list(OFFLINE.glob("t_*.py")) + list(OFFLINE.glob("js_*.py")))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = suites()
    if argv:
        wanted = set(argv)
        selected = [p for p in selected if p.stem in wanted or p.name in wanted]
    if not selected:
        print("no offline suites selected")
        return 2

    failures = []
    for path in selected:
        print(f"\n===== {path.stem} =====", flush=True)
        try:
            proc = subprocess.run([sys.executable, str(path)], cwd=ROOT, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT {path.name} (>120s)", file=sys.stderr)
            failures.append(path.stem)
            continue
        if proc.returncode:
            failures.append(path.stem)

    print(f"\n{len(selected) - len(failures)}/{len(selected)} offline suites passed")
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
