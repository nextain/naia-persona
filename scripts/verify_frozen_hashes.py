#!/usr/bin/env python3
"""Check that every frozen suite still hashes to the constant its runner pins.

Three scripts refuse to run unless a suite's SHA-256 matches a constant compiled
into them, because a suite edited after results exist is no longer held out.
That guard only fires when someone runs the script. This check fires in CI, so a
suite edited by accident is caught at the commit rather than months later.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PAIRS = [
    ("scripts/run_heldout_confirmation.py", "SUITE_SHA256", "examples/naia-v13/identity-confirmation-v3-prompts.json"),
    ("scripts/run_purity_probe.py", "PROBE_SHA256", "benchmark/language-purity-probe.json"),
    ("scripts/run_delegation_probe.py", "PROBE_SHA256", "benchmark/delegation-probe.json"),
]


def main() -> int:
    failures = []
    for script, constant, suite in PAIRS:
        script_path, suite_path = REPO / script, REPO / suite
        if not script_path.exists() or not suite_path.exists():
            failures.append(f"{script} 또는 {suite} 가 없습니다")
            continue
        match = re.search(rf'^{constant}\s*=\s*"([0-9a-f]{{64}})"', script_path.read_text(encoding="utf-8"), re.M)
        if not match:
            failures.append(f"{script} 에서 {constant} 를 찾지 못했습니다")
            continue
        actual = hashlib.sha256(suite_path.read_bytes()).hexdigest()
        if actual != match.group(1):
            failures.append(f"{suite} 의 해시가 바뀌었습니다: {actual}, 스크립트 기대값 {match.group(1)}")
        else:
            print(f"ok  {suite}")
    for line in failures:
        print(f"FAIL  {line}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
