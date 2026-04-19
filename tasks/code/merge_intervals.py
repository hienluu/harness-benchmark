"""Fix an off-by-one bug in merge_intervals."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tasks.registry import Score, Task, Trajectory


BUGGY = '''\
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start >= last_end:  # BUG: touching intervals (start == last_end) should merge
            merged.append((start, end))
        else:
            merged[-1] = (last_start, max(last_end, end))
    return merged
'''

TESTS = '''\
from merge import merge_intervals

def test_basic():
    assert merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]

def test_touching():
    # Touching intervals (e.g., [1,4] and [4,5]) should merge.
    assert merge_intervals([(1, 4), (4, 5)]) == [(1, 5)]

def test_single():
    assert merge_intervals([(1, 5)]) == [(1, 5)]

def test_empty():
    assert merge_intervals([]) == []

def test_nested():
    assert merge_intervals([(1, 10), (2, 3), (4, 5)]) == [(1, 10)]
'''

PROMPT = (
    "There is a buggy function in `merge.py`. Tests in `test_merge.py` fail. "
    "Run the tests, find the bug, fix `merge.py` so all tests pass, then call "
    "`finish` with a one-line summary. Do not modify the tests."
)


def _evaluator(traj: Trajectory, scratch: Path) -> Score:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_merge.py"],
        cwd=str(scratch),
        capture_output=True,
        text=True,
        timeout=60,
    )
    passed = result.returncode == 0
    return Score(
        passed=passed,
        score=1.0 if passed else 0.0,
        details=(result.stdout + result.stderr)[-2000:],
    )


TASK = Task(
    id="code_merge_intervals",
    category="code",
    prompt=PROMPT,
    scratch_files={"merge.py": BUGGY, "test_merge.py": TESTS},
    evaluator=_evaluator,
)
