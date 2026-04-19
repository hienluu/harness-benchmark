"""Implement an LRUCache class from a spec against provided tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tasks.registry import Score, Task, Trajectory


STUB = '''\
class LRUCache:
    """A fixed-capacity cache that evicts least-recently-used entries.

    Required methods:
      - __init__(self, capacity: int)
      - get(self, key) -> value or -1 if missing; marks key as most recent
      - put(self, key, value): inserts or updates, evicts LRU if over capacity
    """
    def __init__(self, capacity):
        raise NotImplementedError

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError
'''

TESTS = '''\
from cache import LRUCache

def test_basic_get_put():
    c = LRUCache(2)
    c.put(1, "a")
    c.put(2, "b")
    assert c.get(1) == "a"
    c.put(3, "c")  # evicts key 2
    assert c.get(2) == -1
    assert c.get(3) == "c"

def test_update_refreshes():
    c = LRUCache(2)
    c.put(1, "a")
    c.put(2, "b")
    c.put(1, "A")  # update, makes 1 most recent
    c.put(3, "c")  # should evict 2, not 1
    assert c.get(1) == "A"
    assert c.get(2) == -1

def test_missing():
    c = LRUCache(1)
    assert c.get(99) == -1

def test_capacity_one():
    c = LRUCache(1)
    c.put(1, "a")
    c.put(2, "b")
    assert c.get(1) == -1
    assert c.get(2) == "b"
'''

PROMPT = (
    "Implement `LRUCache` in `cache.py` against the spec in its docstring. "
    "Tests in `test_cache.py` must all pass. Do not modify the tests. "
    "When done, run pytest to confirm, then call `finish`."
)


def _evaluator(traj: Trajectory, scratch: Path) -> Score:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_cache.py"],
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
    id="code_lru_cache",
    category="code",
    prompt=PROMPT,
    scratch_files={"cache.py": STUB, "test_cache.py": TESTS},
    evaluator=_evaluator,
)
